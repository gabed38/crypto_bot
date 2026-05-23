#!/usr/bin/env python3
"""
Crypto Bot — Quant Screen Backtester
=====================================

Replays the quantitative pre-screen strategy over historical CoinGecko data
to estimate how the screening algorithm would have performed without calling
the LLM.  This is a *signal quality* backtest, not a full trading simulation.

What it tests
-------------
For each date in the lookback window the script:

  1. Fetches the top-N coins by market cap as they stood at that date
     (using the /coins/markets endpoint with a fixed snapshot)
  2. Runs screen_coins() to pick the top-K candidates
  3. Looks forward `hold_days` days and records the price change
  4. Reports aggregate stats: win rate, avg return, Sharpe proxy, sector distribution

Because CoinGecko's free API returns *current* market data (not historical
snapshots), the backtest uses the /coins/{id}/market_chart endpoint to pull
daily OHLC history for each coin and reconstructs the screen retrospectively.

Usage
-----
  python scripts/backtest.py                        # 30-day window, 7d hold
  python scripts/backtest.py --days 60 --hold 3    # 60-day window, 3d hold
  python scripts/backtest.py --top 100 --candidates 20
  python scripts/backtest.py --csv results.csv     # export raw trade log
"""

import sys
import os
import time
import json
import math
import argparse
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.quant_screen import screen_coins, compute_volatility_stats, get_sector


# ── CoinGecko helpers ─────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 3, sleep: float = 1.2) -> Optional[dict]:
    """Simple GET with retry and rate-limit sleep."""
    import requests
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                wait = 60 if attempt == 0 else 120
                print(f"  [rate-limit] sleeping {wait}s…")
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(sleep)
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [error] {url}: {e}")
            time.sleep(sleep * (attempt + 1))
    return None


def fetch_top_coins_now(top_n: int = 100) -> List[Dict]:
    """Fetch current top-N coins from CoinGecko /coins/markets."""
    BASE = "https://api.coingecko.com/api/v3"
    coins = []
    per_page = min(top_n, 250)
    data = _get(f"{BASE}/coins/markets", {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "1h,24h,7d",
    })
    if data:
        coins.extend(data)
    return coins[:top_n]


def fetch_daily_prices(coin_id: str, days: int) -> List[Dict]:
    """
    Fetch daily OHLC for a coin over `days` past days.
    Returns list of {date, open, high, low, close} sorted oldest-first.
    """
    BASE = "https://api.coingecko.com/api/v3"
    data = _get(f"{BASE}/coins/{coin_id}/ohlc", {
        "vs_currency": "usd",
        "days": days,
    }, sleep=0.8)
    if not data:
        return []

    # CoinGecko returns [[timestamp_ms, open, high, low, close], ...]
    result = []
    for row in data:
        if len(row) >= 5:
            ts_ms, o, h, l, c = row[0], row[1], row[2], row[3], row[4]
            date = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            result.append({"date": date, "open": o, "high": h, "low": l, "close": c})

    # Dedupe by date (keep last candle per day)
    seen = {}
    for row in result:
        seen[row["date"]] = row
    return sorted(seen.values(), key=lambda r: r["date"])


def build_simulated_coin(coin: Dict, ohlc_history: List[Dict], as_of_date: str) -> Optional[Dict]:
    """
    Build a fake coin dict representing how `coin` looked on `as_of_date`.

    Uses the OHLC record closest to (but not after) as_of_date as the
    reference candle, and computes 24h / 7d changes from history.
    Returns None if there is insufficient data.
    """
    if not ohlc_history:
        return None

    # Find index of the as_of_date candle
    dates = [r["date"] for r in ohlc_history]
    try:
        idx = dates.index(as_of_date)
    except ValueError:
        # Pick the closest date before as_of_date
        before = [i for i, d in enumerate(dates) if d <= as_of_date]
        if not before:
            return None
        idx = before[-1]

    ref = ohlc_history[idx]
    ref_close = ref["close"]
    if not ref_close:
        return None

    # 24h change
    change_24h = 0.0
    if idx >= 1:
        prev = ohlc_history[idx - 1]["close"]
        if prev:
            change_24h = (ref_close - prev) / prev * 100

    # 7d change
    change_7d = 0.0
    if idx >= 7:
        prev7 = ohlc_history[idx - 7]["close"]
        if prev7:
            change_7d = (ref_close - prev7) / prev7 * 100

    # 1h change: unavailable from daily OHLC — approximate as open→close / 2
    change_1h = (ref_close - ref["open"]) / ref["open"] * 100 / 2 if ref["open"] else 0

    sim = dict(coin)
    sim["current_price"] = ref_close
    sim["price_change_percentage_24h"] = change_24h
    sim["price_change_percentage_7d_in_currency"] = change_7d
    sim["price_change_percentage_1h_in_currency"] = change_1h
    # Volume and market cap are kept from current data (imperfect but unavoidable
    # without a premium API)

    return sim


# ── Backtest core ──────────────────────────────────────────────────────────────

def run_backtest(
    lookback_days: int = 30,
    hold_days: int = 7,
    top_n: int = 100,
    max_candidates: int = 15,
    min_volume_usd: float = 50_000_000,
    min_stop_multiple: float = 1.5,
    step_days: int = 7,
    csv_path: Optional[str] = None,
) -> None:
    """
    Run the quant screen backtest.

    Args:
        lookback_days:   How many past days to cover (each step_days days apart).
        hold_days:       Forward return window to measure per screened trade.
        top_n:           Coins to fetch from CoinGecko (universe size).
        max_candidates:  Candidates returned by screen_coins per step.
        min_volume_usd:  Volume floor for the screen.
        min_stop_multiple: Volatility filter threshold.
        step_days:       How many days between each simulated screen run.
        csv_path:        If set, write a CSV trade log to this path.
    """
    print("\n" + "=" * 70)
    print("CRYPTO BOT — QUANT SCREEN BACKTEST")
    print("=" * 70)
    print(f"  Universe      : top {top_n} coins")
    print(f"  Lookback      : {lookback_days} days")
    print(f"  Hold period   : {hold_days} days")
    print(f"  Screen freq   : every {step_days} days")
    print(f"  Max candidates: {max_candidates}")
    print(f"  Volume floor  : ${min_volume_usd / 1e6:.0f}M")
    print(f"  Min stop×     : {min_stop_multiple}x")
    print()

    # --- Step 1: Fetch current top-N coins (universe) ---
    print("  [1/4] Fetching coin universe from CoinGecko...")
    coins = fetch_top_coins_now(top_n)
    if not coins:
        print("  ERROR: could not fetch coin data")
        return
    print(f"  Fetched {len(coins)} coins")

    # --- Step 2: Fetch OHLC history for every coin ---
    # Need lookback + hold_days to measure forward returns
    total_days_needed = lookback_days + hold_days + 10  # buffer
    print(f"\n  [2/4] Fetching {total_days_needed}d OHLC history for {len(coins)} coins…")
    print(f"        (this takes ~{len(coins) * 1:.0f}s due to rate limits)")

    ohlc_cache: Dict[str, List[Dict]] = {}
    for i, coin in enumerate(coins, 1):
        cid = coin["id"]
        hist = fetch_daily_prices(cid, days=total_days_needed)
        ohlc_cache[cid] = hist
        if i % 10 == 0:
            print(f"        {i}/{len(coins)} fetched…")

    print(f"  History fetched for {len(ohlc_cache)} coins")

    # --- Step 3: Replay screen across dates ---
    print(f"\n  [3/4] Replaying quant screen across date range…")

    today = datetime.utcnow().date()
    step_dates = []
    d = today - timedelta(days=lookback_days)
    while d <= today - timedelta(days=hold_days):
        step_dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=step_days)

    print(f"  Simulation dates: {len(step_dates)} steps")

    all_trades: List[Dict] = []

    for date_str in step_dates:
        # Build simulated coin snapshots for this date
        sim_coins = []
        for coin in coins:
            hist = ohlc_cache.get(coin["id"], [])
            sim = build_simulated_coin(coin, hist, date_str)
            if sim:
                sim_coins.append(sim)

        if not sim_coins:
            continue

        # Run the screen
        screened = screen_coins(
            coins=sim_coins,
            trending_ids=set(),
            held_coin_ids=set(),
            min_volume_usd=min_volume_usd,
            max_candidates=max_candidates,
            fear_greed_value=50,  # neutral — we don't have historical F&G
        )

        # Volatility filter
        accepted = []
        for coin in screened:
            hist = ohlc_cache.get(coin["id"], [])
            # Get prices up to this date
            dates = [r["date"] for r in hist]
            try:
                idx = dates.index(date_str)
            except ValueError:
                before = [i for i, d in enumerate(dates) if d <= date_str]
                idx = before[-1] if before else -1
            if idx < 0:
                accepted.append(coin)
                continue
            price_history = [r["close"] for r in hist[max(0, idx - 14):idx + 1]]
            vol_stats = compute_volatility_stats(price_history)
            coin.update(vol_stats)
            sm = vol_stats.get("stop_multiple")
            if sm is None or sm >= min_stop_multiple:
                accepted.append(coin)

        # Measure forward return for each accepted coin
        forward_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=hold_days)).strftime("%Y-%m-%d")

        for coin in accepted:
            cid = coin["id"]
            hist = ohlc_cache.get(cid, [])
            dates_map = {r["date"]: r["close"] for r in hist}

            entry_price = coin.get("current_price")
            exit_price = dates_map.get(forward_date)

            # If exact forward date missing, find nearest
            if exit_price is None:
                future_dates = sorted(d for d in dates_map if d > date_str)
                close_dates = [d for d in future_dates if d <= forward_date]
                if close_dates:
                    exit_price = dates_map[close_dates[-1]]

            fwd_return = None
            if entry_price and exit_price and entry_price > 0:
                fwd_return = (exit_price - entry_price) / entry_price * 100

            all_trades.append({
                "screen_date": date_str,
                "forward_date": forward_date,
                "coin_id": cid,
                "symbol": coin.get("symbol", "?").upper(),
                "sector": get_sector(cid),
                "screen_score": coin.get("screen_score"),
                "momentum": coin.get("momentum"),
                "rs_vs_btc": coin.get("rs_vs_btc"),
                "vol_signal": coin.get("vol_signal"),
                "stop_multiple": coin.get("stop_multiple"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "fwd_return_pct": round(fwd_return, 2) if fwd_return is not None else None,
                "hit_stop_loss": fwd_return is not None and fwd_return <= -15.0,
            })

    # --- Step 4: Report results ---
    print(f"\n  [4/4] Analysing {len(all_trades)} simulated trades…\n")

    _print_report(all_trades, hold_days)

    if csv_path:
        _write_csv(all_trades, csv_path)
        print(f"\n  Trade log written to {csv_path}")


def _print_report(trades: List[Dict], hold_days: int) -> None:
    """Print a formatted results report to stdout."""

    measurable = [t for t in trades if t["fwd_return_pct"] is not None]
    if not measurable:
        print("  No measurable trades (could not compute forward returns).")
        return

    returns = [t["fwd_return_pct"] for t in measurable]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    stop_losses = [t for t in measurable if t["hit_stop_loss"]]

    avg_ret = sum(returns) / len(returns)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    win_rate = len(wins) / len(returns) * 100
    stop_rate = len(stop_losses) / len(measurable) * 100

    # Sharpe proxy: mean / std of returns (annualised naive)
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance)
        # Periods per year based on hold_days
        periods_per_year = 365 / hold_days
        sharpe = (avg_ret / std_r) * math.sqrt(periods_per_year) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    sep = "─" * 60

    print(sep)
    print(f"  OVERALL ({len(measurable)} screened picks, {hold_days}d forward return)")
    print(sep)
    print(f"  Win rate          : {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg return        : {avg_ret:+.2f}%")
    print(f"  Avg win           : {avg_win:+.2f}%")
    print(f"  Avg loss          : {avg_loss:+.2f}%")
    print(f"  Best trade        : {max(returns):+.2f}%")
    print(f"  Worst trade       : {min(returns):+.2f}%")
    print(f"  Stop-loss hit rate: {stop_rate:.1f}%  ({len(stop_losses)} trades down ≥15%)")
    print(f"  Sharpe proxy      : {sharpe:+.2f}  (annualised, naive)")

    # Breakdown by sector
    from collections import defaultdict
    sector_returns: Dict[str, List[float]] = defaultdict(list)
    for t in measurable:
        sector_returns[t["sector"]].append(t["fwd_return_pct"])

    print(f"\n  BY SECTOR")
    print(sep)
    print(f"  {'Sector':<14} {'Picks':>6} {'Win%':>7} {'Avg Ret':>10}")
    print(f"  {'──────':<14} {'─────':>6} {'────':>7} {'───────':>10}")
    for sector, rets in sorted(sector_returns.items(), key=lambda x: -len(x[1])):
        wr = sum(1 for r in rets if r > 0) / len(rets) * 100
        ar = sum(rets) / len(rets)
        print(f"  {sector:<14} {len(rets):>6} {wr:>6.1f}% {ar:>+9.2f}%")

    # Breakdown by vol signal
    vol_returns: Dict[str, List[float]] = defaultdict(list)
    for t in measurable:
        sig = t.get("vol_signal") or "UNKNOWN"
        vol_returns[sig].append(t["fwd_return_pct"])

    print(f"\n  BY VOLATILITY SIGNAL")
    print(sep)
    print(f"  {'Signal':<12} {'Picks':>6} {'Win%':>7} {'Avg Ret':>10}")
    print(f"  {'──────':<12} {'─────':>6} {'────':>7} {'───────':>10}")
    for sig in ["LOW", "MEDIUM", "HIGH", "EXTREME", "UNKNOWN"]:
        rets = vol_returns.get(sig, [])
        if not rets:
            continue
        wr = sum(1 for r in rets if r > 0) / len(rets) * 100
        ar = sum(rets) / len(rets)
        print(f"  {sig:<12} {len(rets):>6} {wr:>6.1f}% {ar:>+9.2f}%")

    # Top 10 coins by frequency
    from collections import Counter
    freq = Counter(t["symbol"] for t in measurable)
    print(f"\n  TOP 10 MOST SCREENED COINS")
    print(sep)
    for sym, count in freq.most_common(10):
        coin_returns = [t["fwd_return_pct"] for t in measurable if t["symbol"] == sym]
        ar = sum(coin_returns) / len(coin_returns)
        wr = sum(1 for r in coin_returns if r > 0) / len(coin_returns) * 100
        print(f"  {sym:<8} screened {count:>3}×  avg={ar:>+7.2f}%  win={wr:.0f}%")

    print(f"\n  {'─' * 60}")


def _write_csv(trades: List[Dict], path: str) -> None:
    """Write the full trade log to a CSV file."""
    if not trades:
        return
    fieldnames = list(trades[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backtest the crypto bot quant screen strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days",       type=int,   default=30,          help="Lookback window (days)")
    parser.add_argument("--hold",       type=int,   default=7,           help="Forward return hold period (days)")
    parser.add_argument("--top",        type=int,   default=100,         help="Universe size (top N coins)")
    parser.add_argument("--candidates", type=int,   default=15,          help="Max candidates per screen")
    parser.add_argument("--volume",     type=float, default=50_000_000,  help="Min 24h volume filter (USD)")
    parser.add_argument("--stop-mult",  type=float, default=1.5,         help="Min stop/vol ratio")
    parser.add_argument("--step",       type=int,   default=7,           help="Days between screen runs")
    parser.add_argument("--csv",        type=str,   default=None,        help="Export trade log to CSV")
    args = parser.parse_args()

    run_backtest(
        lookback_days=args.days,
        hold_days=args.hold,
        top_n=args.top,
        max_candidates=args.candidates,
        min_volume_usd=args.volume,
        min_stop_multiple=args.stop_mult,
        step_days=args.step,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
