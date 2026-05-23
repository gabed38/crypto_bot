#!/usr/bin/env python3
"""
Coin Profile Builder

Fetches 90-day daily price history for the top 150 coins and computes
behavioral statistics for each: volatility, BTC correlation, beta,
momentum persistence, drawdown, and regime-conditioned 7-day forward returns.

Profiles are saved to data/coin_profiles/profiles.json and injected into
the LLM trade-analysis prompt as context for every screening session.

Run weekly (or manually before a trading session):
    python scripts/build_coin_profiles.py

Flags:
    --coins N        Override how many coins to profile (default 150)
    --days N         Override history window in days (default 90)
    --sleep N        Seconds to sleep between CoinGecko calls (default 2.0)
    --coin-ids a,b   Profile only specific comma-separated coin IDs
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from loguru import logger

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_FILE = REPO_ROOT / "data" / "coin_profiles" / "profiles.json"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# ── HTTP session ───────────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update({"Accept": "application/json"})


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def _get(url: str, params: dict = None, retries: int = 3):
    """GET with exponential backoff on 429 rate-limit responses."""
    for attempt in range(retries):
        try:
            resp = _session.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                wait = 30 * (2 ** attempt)  # 30s → 60s → 120s
                logger.warning(
                    f"429 rate-limited — waiting {wait}s "
                    f"(attempt {attempt + 1}/{retries})"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(10)
                logger.warning(f"Retry {attempt + 1}/{retries}: {e}")
            else:
                logger.error(f"All {retries} attempts failed for {url}: {e}")
    return None


def fetch_top_coins(n: int = 150) -> list:
    """Fetch the top N coins by market cap from CoinGecko /coins/markets."""
    coins = []
    per_page = 50
    pages = math.ceil(n / per_page)
    for page in range(1, pages + 1):
        data = _get(
            f"{COINGECKO_BASE}/coins/markets",
            {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
            },
        )
        if data:
            coins.extend(data)
        time.sleep(2.0)
    logger.info(f"Fetched metadata for {len(coins)} coins")
    return coins[:n]


def fetch_closing_prices(coin_id: str, days: int = 90) -> list:
    """
    Fetch daily closing prices via /coins/{id}/market_chart.

    Returns a list of (date_str, close_price) tuples sorted oldest-first,
    deduplicated by date (keeps last price for each calendar day).
    """
    data = _get(
        f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
        {"vs_currency": "usd", "days": days, "interval": "daily"},
    )
    if not data:
        return []
    by_date = {}
    for ts_ms, price in data.get("prices", []):
        d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        by_date[d] = float(price)
    return sorted(by_date.items())  # [(date, price), ...] oldest first


def fetch_fg_history(days: int = 90) -> dict:
    """
    Fetch historical Fear & Greed Index from alternative.me.

    Returns {date_str: fg_int} for up to `days` past days.
    """
    data = _get(
        "https://api.alternative.me/fng/",
        {"limit": days, "format": "json"},
    )
    if not data:
        return {}
    fg_by_date = {}
    for entry in data.get("data", []):
        ts = int(entry.get("timestamp", 0))
        val = int(entry.get("value", 50))
        if ts:
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            fg_by_date[d] = val
    return fg_by_date


# ══════════════════════════════════════════════════════════════════════════════
# PROFILE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _pearson(xs: list, ys: list) -> float:
    """Pearson correlation between two equal-length lists."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(n - 1, 1)
    vx = sum((x - mx) ** 2 for x in xs) / max(n - 1, 1)
    vy = sum((y - my) ** 2 for y in ys) / max(n - 1, 1)
    denom = math.sqrt(vx) * math.sqrt(vy)
    return cov / denom if denom > 0 else 0.0


def compute_profile(
    coin_id: str,
    symbol: str,
    name: str,
    coin_closes: list,       # [(date, price), ...]
    btc_closes: list,        # [(date, price), ...]
    fg_by_date: dict,        # {date: fg_int}
) -> dict | None:
    """
    Compute a behavioral profile for one coin using 90 days of closing prices.

    Metrics:
      - daily_vol_pct         — std dev of daily log returns (annualised noise)
      - btc_corr              — Pearson correlation of daily returns with BTC
      - btc_beta              — cov(coin, btc) / var(btc)
      - momentum_persistence  — lag-1 autocorrelation of daily returns
                                (positive = trendy, negative = mean-reverting)
      - worst_drawdown_pct    — worst peak-to-trough over the period
      - avg_drawdown_pct      — average drawdown from running peak (daily)
      - regime_returns        — 7-day forward avg return and win rate, grouped
                                by the Fear & Greed band on the entry day

    Returns None if there is insufficient data.
    """
    if len(coin_closes) < 10 or len(btc_closes) < 10:
        return None

    coin_by_date = dict(coin_closes)
    btc_by_date = dict(btc_closes)

    # Dates where both coin and BTC have prices
    common = sorted(set(coin_by_date) & set(btc_by_date))
    if len(common) < 10:
        return None

    # Daily log returns for both
    coin_rets, btc_rets = [], []
    for i in range(1, len(common)):
        d, d_prev = common[i], common[i - 1]
        cp, cpp = coin_by_date.get(d), coin_by_date.get(d_prev)
        bp, bpp = btc_by_date.get(d), btc_by_date.get(d_prev)
        if cp and cpp and bp and bpp and cpp > 0 and bpp > 0:
            coin_rets.append(math.log(cp / cpp))
            btc_rets.append(math.log(bp / bpp))

    if len(coin_rets) < 5:
        return None

    n = len(coin_rets)
    mean_c = sum(coin_rets) / n

    # Volatility: std dev of daily log returns → convert to %
    var_c = sum((r - mean_c) ** 2 for r in coin_rets) / max(n - 1, 1)
    daily_vol_pct = math.sqrt(var_c) * 100

    # BTC correlation and beta
    mean_b = sum(btc_rets) / n
    var_b = sum((r - mean_b) ** 2 for r in btc_rets) / max(n - 1, 1)
    cov = (
        sum((c - mean_c) * (b - mean_b) for c, b in zip(coin_rets, btc_rets))
        / max(n - 1, 1)
    )
    corr = cov / (math.sqrt(var_c) * math.sqrt(var_b)) if var_c > 0 and var_b > 0 else 0.0
    beta = cov / var_b if var_b > 0 else 1.0

    # Momentum persistence: lag-1 autocorrelation of coin returns
    # Positive → yesterday's return predicts today's (trending)
    # Negative → yesterday's return reverses (mean-reverting)
    momentum_persistence = _pearson(coin_rets[:-1], coin_rets[1:]) if n >= 6 else 0.0

    # Drawdown from running peak (daily)
    prices = [coin_by_date[d] for d in common]
    peak = prices[0]
    drawdowns = []
    for p in prices:
        peak = max(peak, p)
        drawdowns.append((p - peak) / peak * 100)
    worst_drawdown = min(drawdowns)
    avg_drawdown = sum(drawdowns) / len(drawdowns)

    # Regime-conditioned 7-day forward returns
    # Bucket entries by F&G value on entry day, compute forward 7d return
    buckets: dict = {
        "extreme_fear": [],
        "fear": [],
        "neutral": [],
        "greed": [],
        "extreme_greed": [],
    }
    for i, d in enumerate(common):
        if d not in fg_by_date:
            continue
        fwd_idx = i + 7
        if fwd_idx >= len(common):
            continue
        p_now = coin_by_date.get(d)
        p_fwd = coin_by_date.get(common[fwd_idx])
        if not p_now or not p_fwd or p_now <= 0:
            continue
        fwd_pct = (p_fwd - p_now) / p_now * 100
        fg = fg_by_date[d]
        if fg <= 20:
            buckets["extreme_fear"].append(fwd_pct)
        elif fg <= 40:
            buckets["fear"].append(fwd_pct)
        elif fg <= 60:
            buckets["neutral"].append(fwd_pct)
        elif fg <= 80:
            buckets["greed"].append(fwd_pct)
        else:
            buckets["extreme_greed"].append(fwd_pct)

    regime_returns = {}
    for bucket, rets in buckets.items():
        if rets:
            avg = sum(rets) / len(rets)
            wins = sum(1 for r in rets if r > 0)
            regime_returns[bucket] = {
                "n": len(rets),
                "avg_7d_pct": round(avg, 1),
                "win_rate": round(wins / len(rets), 2),
            }
        else:
            regime_returns[bucket] = {"n": 0, "avg_7d_pct": None, "win_rate": None}

    return {
        "symbol": symbol.upper(),
        "name": name,
        "data_days": len(common),
        "daily_vol_pct": round(daily_vol_pct, 2),
        "btc_corr": round(corr, 3),
        "btc_beta": round(beta, 2),
        "momentum_persistence": round(momentum_persistence, 3),
        "worst_drawdown_pct": round(worst_drawdown, 1),
        "avg_drawdown_pct": round(avg_drawdown, 1),
        "regime_returns": regime_returns,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build coin behavioral profiles")
    parser.add_argument("--coins", type=int, default=150,
                        help="Number of top coins to profile (default: 150)")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of history to fetch (default: 90)")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Seconds to sleep between CoinGecko calls (default: 2.0)")
    parser.add_argument("--coin-ids", type=str, default="",
                        help="Comma-separated list of specific coin IDs to profile")
    args = parser.parse_args()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("COIN PROFILE BUILDER")
    print(f"  Coins: {args.coins}  |  History: {args.days}d  |  Sleep: {args.sleep}s")
    print("=" * 65)

    # ── Step 1: Fear & Greed history ─────────────────────────────────────────
    print(f"\n[1/4] Fetching Fear & Greed history ({args.days} days)...")
    fg_by_date = fetch_fg_history(args.days)
    print(f"  Got F&G data for {len(fg_by_date)} dates")
    if fg_by_date:
        sample_dates = sorted(fg_by_date)
        print(f"  Range: {sample_dates[0]} → {sample_dates[-1]}")

    # ── Step 2: Coin universe ─────────────────────────────────────────────────
    if args.coin_ids:
        # Manual override: build minimal metadata list from explicit IDs
        id_list = [x.strip() for x in args.coin_ids.split(",") if x.strip()]
        top_coins = [{"id": cid, "symbol": cid, "name": cid, "market_cap_rank": "?"}
                     for cid in id_list]
        print(f"\n[2/4] Using {len(top_coins)} explicit coin IDs: {', '.join(id_list)}")
    else:
        print(f"\n[2/4] Fetching top {args.coins} coins metadata...")
        top_coins = fetch_top_coins(args.coins)
        print(f"  Fetched {len(top_coins)} coins")

    # ── Step 3: BTC price history (correlation baseline) ──────────────────────
    print("\n[3/4] Fetching BTC price history (correlation baseline)...")
    btc_closes = fetch_closing_prices("bitcoin", args.days)
    print(f"  BTC: {len(btc_closes)} daily data points")
    time.sleep(args.sleep)

    if not btc_closes:
        print("ERROR: Could not fetch BTC price history — cannot compute correlations.")
        sys.exit(1)

    # ── Step 4: Per-coin profile computation ──────────────────────────────────
    print(f"\n[4/4] Computing profiles for {len(top_coins)} coins...")

    # Load existing profiles so we can merge (incremental update)
    existing = {}
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE) as f:
                existing = json.load(f).get("profiles", {})
        except Exception:
            pass

    profiles = {}
    failed = []

    for i, coin in enumerate(top_coins):
        coin_id = coin.get("id", "")
        symbol = coin.get("symbol", "?")
        name = coin.get("name", "?")
        rank = coin.get("market_cap_rank", "?")

        print(
            f"  [{i + 1:>3}/{len(top_coins)}] "
            f"#{str(rank):<5} {symbol.upper():<8} {coin_id:<30}",
            end="",
            flush=True,
        )

        closes = fetch_closing_prices(coin_id, args.days)
        time.sleep(args.sleep)

        if not closes:
            print("⚠  no price data")
            failed.append(coin_id)
            continue

        profile = compute_profile(coin_id, symbol, name, closes, btc_closes, fg_by_date)
        if profile:
            profiles[coin_id] = profile
            rr = profile["regime_returns"]
            # Show regime summary in one line
            parts = []
            for band, label in [
                ("extreme_fear", "XF"), ("fear", "F"),
                ("neutral", "N"), ("greed", "G"), ("extreme_greed", "XG"),
            ]:
                r = rr.get(band, {})
                if r.get("n", 0) > 0 and r.get("avg_7d_pct") is not None:
                    sign = "+" if r["avg_7d_pct"] >= 0 else ""
                    parts.append(f"{label}:{sign}{r['avg_7d_pct']:.0f}%")
            regime_str = " ".join(parts) if parts else "no regime data"
            print(
                f"✓  vol={profile['daily_vol_pct']:.1f}%/d  "
                f"corr={profile['btc_corr']:.2f}  beta={profile['btc_beta']:.1f}x  "
                f"[{regime_str}]"
            )
        else:
            print("⚠  insufficient data for profile")
            failed.append(coin_id)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "history_days": args.days,
        "coin_count": len(profiles),
        "profiles": profiles,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"Done. {len(profiles)} profiles saved to:")
    print(f"  {OUTPUT_FILE}")
    if failed:
        failed_str = ", ".join(failed[:15])
        if len(failed) > 15:
            failed_str += f" ... (+{len(failed) - 15} more)"
        print(f"Failed ({len(failed)}): {failed_str}")


if __name__ == "__main__":
    main()
