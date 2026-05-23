#!/usr/bin/env python3
"""
Unified Intraday Crypto Trading Script

Designed to run every 2 hours via cron.

Executes three phases in sequence:

  Phase 1 — Position Management
    - Load all open positions and fetch current prices
    - Check vol-adaptive stop-losses (3× daily vol, 10–25% per coin)
    - Run LLM hold/close analysis on remaining positions
    - Close positions flagged by stop-loss or LLM

  Phase 2 — New Trade Discovery
    - Fetch top crypto prices, trending coins, Fear & Greed Index
    - Fetch crypto news from RSS feeds
    - Run LLM analysis with market data + news + lessons
    - Apply conviction / risk filters and record qualifying trades

  Phase 3 — Post-Trade Analysis
    - If any positions were closed in Phase 1, run LLM review
    - Extract lessons and persist to lessons.json for next run
"""

import sys
import os
import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv(REPO_ROOT / ".env", override=True)

# ── imports ──────────────────────────────────────────────────────────────────
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data_ingestion.crypto_prices import CryptoPriceClient
from src.data_ingestion.crypto_news import CryptoNewsClient
from src.analysis.llm_analyzer import LLMAnalyzer
from src.analysis.pnl_tracker import PnLTracker
from src.analysis.quant_screen import (
    screen_coins, format_screen_summary, enrich_with_volatility,
    check_sector_concentration, get_market_regime,
)
from src.analysis.coin_profiles import load_profiles, format_profiles_for_llm
from src.trading.executor import TradeExecutor
from src.trading.risk import RiskManager

from helpers import (
    load_open_positions,
    save_open_positions,
    add_to_open_positions,
    append_resolved_trade,
    load_recent_resolved,
    evaluate_position,
    check_time_horizon_expired,
    check_mid_horizon_stale,
    save_lessons,
    load_recent_lessons,
    POSITIONS_FILE,
    RESOLVED_FILE,
    LESSONS_FILE,
)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — POSITION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1_position_management(
    price_client: CryptoPriceClient,
    llm_analyzer: LLMAnalyzer,  # kept in signature for Phase 3 post-trade analysis
    trade_executor: TradeExecutor,
    risk_manager: RiskManager,
    config: Config,
    logger,
) -> List[Dict]:
    """
    Phase 1: Check every open position and close any that warrant it.

    All exits are purely price-based — no LLM call.  Four mechanical rules
    fire in priority order:

      1. STOP_LOSS         — price dropped below vol-adaptive stop (3× daily vol, 10–25%)
      2. TRAILING_STOP     — high-water mark trailing stop crossed
      3. TAKE_PROFIT       — price reached the target % set at entry
      4. PROFIT_PROTECTION — peak gain ≥ 4% then gave back > 50% of that peak
      5. TIME_EXPIRED      — backstop: position held past its time horizon

    Returns list of close records for positions that were actually closed.
    """
    print("\n" + "=" * 70)
    print("PHASE 1 — POSITION MANAGEMENT")
    print("=" * 70)

    positions = load_open_positions()
    if not positions:
        print("  No open positions to check.")
        return []

    print(f"\n  [1/2] Fetching current prices for {len(positions)} open positions...")

    # Batch fetch current prices for all held coins
    coin_ids = list({p.get("coin_id", "") for p in positions if p.get("coin_id")})
    current_prices = {}
    if coin_ids:
        try:
            from requests import Session
            session = Session()
            resp = session.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": ",".join(coin_ids), "vs_currencies": "usd"},
                timeout=15,
            )
            resp.raise_for_status()
            for cid, data in resp.json().items():
                current_prices[cid] = data.get("usd")
        except Exception as e:
            logger.error(f"Failed to batch fetch prices: {e}")

    # Evaluate each position and update trailing high-water marks
    evaluated = []
    for pos in positions:
        coin_id = pos.get("coin_id", "")
        price = current_prices.get(coin_id)
        result = evaluate_position(pos, price)
        # Update trailing high-water mark while we have a fresh price
        if result["status"] == "open" and price is not None:
            result = risk_manager.update_trailing_high(result, price)
        evaluated.append(result)

    price_ok = [r for r in evaluated if r["status"] == "open"]
    price_fail = [r for r in evaluated if r["status"] == "price_unavailable"]

    print(f"  Priced: {len(price_ok)}  |  Price unavailable: {len(price_fail)}")

    # --- Step 2: Price-based exit rules ---
    # All exits are mechanical — no LLM needed.  Rules checked in priority order:
    #   1. STOP_LOSS         — price fell below adaptive vol-stop from entry
    #   2. TRAILING_STOP     — price pulled back from high-water mark past stop distance
    #   3. TAKE_PROFIT       — price hit the target % set at entry
    #   4. PROFIT_PROTECTION — peak gain ≥ 4%, then gave back > 50% of that peak
    #   5. TIME_EXPIRED      — held past time horizon with no resolution (backstop)
    print(f"\n  [2/2] Checking price-based exit rules...")
    stop_loss_ids: set = set()
    trailing_stop_ids: set = set()
    take_profit_ids: set = set()
    profit_protect_ids: set = set()
    stale_ids: set = set()
    expired_ids: set = set()

    for r in price_ok:
        cid = r.get("coin_id")
        price = r.get("latest_price", 0)
        pnl_pct = r.get("pnl_pct") or 0
        symbol = r.get("symbol", "?").upper()
        target = r.get("target_pct") or 0
        entry = r.get("entry_price", 0)

        if risk_manager.check_stop_loss(r, price):
            stop_loss_ids.add(cid)
        elif risk_manager.check_trailing_stop(r, price):
            trailing_stop_ids.add(cid)
        elif risk_manager.check_take_profit(r, price):
            take_profit_ids.add(cid)
            logger.info(f"Take-profit: {symbol} @ {pnl_pct:+.1f}% (target {target:.1f}%)")
        elif risk_manager.check_profit_protection(r, price):
            profit_protect_ids.add(cid)
        elif check_mid_horizon_stale(r):
            stale_ids.add(cid)
            logger.info(
                f"Mid-horizon stale: {symbol} "
                f"(horizon: {r.get('time_horizon', '?')}, P&L: {pnl_pct:+.1f}% — cutting early)"
            )
        elif check_time_horizon_expired(r):
            expired_ids.add(cid)
            logger.info(
                f"Time horizon expired: {symbol} "
                f"(horizon: {r.get('time_horizon', '?')}, P&L: {pnl_pct:+.1f}%)"
            )

    # Print summary of triggered rules
    rule_summary = []
    if stop_loss_ids:
        syms = ", ".join(r.get("symbol","?").upper() for r in price_ok if r.get("coin_id") in stop_loss_ids)
        rule_summary.append(f"Stop-loss: {syms}")
    if trailing_stop_ids:
        syms = ", ".join(r.get("symbol","?").upper() for r in price_ok if r.get("coin_id") in trailing_stop_ids)
        rule_summary.append(f"Trailing stop: {syms}")
    if take_profit_ids:
        syms = ", ".join(r.get("symbol","?").upper() for r in price_ok if r.get("coin_id") in take_profit_ids)
        rule_summary.append(f"Take-profit: {syms}")
    if profit_protect_ids:
        syms = ", ".join(r.get("symbol","?").upper() for r in price_ok if r.get("coin_id") in profit_protect_ids)
        rule_summary.append(f"Profit protection: {syms}")
    if stale_ids:
        syms = ", ".join(r.get("symbol","?").upper() for r in price_ok if r.get("coin_id") in stale_ids)
        rule_summary.append(f"Mid-horizon stale: {syms}")
    if expired_ids:
        syms = ", ".join(r.get("symbol","?").upper() for r in price_ok if r.get("coin_id") in expired_ids)
        rule_summary.append(f"Time expired: {syms}")

    if rule_summary:
        for line in rule_summary:
            print(f"  {line}")
    else:
        print(f"  All {len(price_ok)} positions holding — no exit rules triggered")

    auto_close_ids = stop_loss_ids | trailing_stop_ids | take_profit_ids | profit_protect_ids | stale_ids | expired_ids

    # --- Execute closes ---
    closed_records: List[Dict] = []
    remaining_open = []

    for r in evaluated:
        coin_id = r.get("coin_id", "")

        if coin_id in stop_loss_ids:
            close_record = _build_close_record(r, "STOP_LOSS", "Stop-loss threshold breached")
            append_resolved_trade(close_record)
            trade_executor.close_position(r)
            closed_records.append(close_record)
            pnl_str = f"{close_record.get('pnl_pct', 0):+.1f}%" if close_record.get("pnl_pct") is not None else "N/A"
            print(f"  [STOP-LOSS] {r.get('symbol', '?').upper()} | P&L: {pnl_str}")

        elif coin_id in trailing_stop_ids:
            pnl_pct = r.get("pnl_pct") or 0
            high = r.get("highest_price") or r.get("lowest_price")
            high_str = f"${high:,.4f}" if high else "?"
            close_record = _build_close_record(
                r, "TRAILING_STOP",
                f"Trailing stop hit — high-water mark {high_str}, stop={r.get('trailing_stop_price', '?')}"
            )
            append_resolved_trade(close_record)
            trade_executor.close_position(r)
            closed_records.append(close_record)
            pnl_str = f"{pnl_pct:+.1f}%"
            print(f"  [TRAILING STOP] {r.get('symbol', '?').upper()} | P&L: {pnl_str} | HWM: {high_str}")

        elif coin_id in take_profit_ids:
            pnl_pct = r.get("pnl_pct") or 0
            target = r.get("target_pct") or 0
            close_record = _build_close_record(
                r, "TAKE_PROFIT",
                f"Price target reached: +{pnl_pct:.1f}% (target was {target:.1f}%)"
            )
            append_resolved_trade(close_record)
            trade_executor.close_position(r)
            closed_records.append(close_record)
            print(f"  [TAKE PROFIT] {r.get('symbol', '?').upper()} | P&L: {pnl_pct:+.1f}%")

        elif coin_id in profit_protect_ids:
            pnl_pct = r.get("pnl_pct") or 0
            entry = float(r.get("entry_price") or 0)
            highest = float(r.get("highest_price") or entry)
            peak_pct = (highest - entry) / entry * 100 if entry else 0
            close_record = _build_close_record(
                r, "PROFIT_PROTECTION",
                f"Peak gain {peak_pct:.1f}% reversed to {pnl_pct:.1f}% — locking in remaining profit"
            )
            append_resolved_trade(close_record)
            trade_executor.close_position(r)
            closed_records.append(close_record)
            print(f"  [PROFIT PROTECTION] {r.get('symbol', '?').upper()} | Peak: {peak_pct:+.1f}% | Now: {pnl_pct:+.1f}%")

        elif coin_id in stale_ids:
            pnl_pct = r.get("pnl_pct") or 0
            horizon = r.get("time_horizon", "?")
            close_record = _build_close_record(
                r, "STALE_POSITION",
                f"Halfway through {horizon} horizon with only {pnl_pct:+.1f}% — thesis not materialising, cutting early"
            )
            append_resolved_trade(close_record)
            trade_executor.close_position(r)
            closed_records.append(close_record)
            print(f"  [STALE / EARLY EXIT] {r.get('symbol', '?').upper()} | Horizon: {horizon} | P&L: {pnl_pct:+.1f}%")

        elif coin_id in expired_ids:
            pnl_pct = r.get("pnl_pct") or 0
            result_label = "TAKE PROFIT" if pnl_pct >= 0 else "CUT LOSS"
            close_record = _build_close_record(
                r, "TIME_EXPIRED",
                f"Time horizon elapsed ({r.get('time_horizon', '?')}) | P&L: {pnl_pct:+.1f}%"
            )
            append_resolved_trade(close_record)
            trade_executor.close_position(r)
            closed_records.append(close_record)
            pnl_str = f"{pnl_pct:+.1f}%"
            print(f"  [TIME EXPIRED / {result_label}] {r.get('symbol', '?').upper()} | P&L: {pnl_str}")

        else:
            # Still open — persist updated price, P&L, and trailing high-water mark
            entry = {k: v for k, v in r.items() if k not in ("status", "trade_result")}
            if r.get("latest_price") is not None:
                entry["latest_price"] = r["latest_price"]
                entry["pnl_pct"] = r.get("pnl_pct")
                entry["pnl_usd"] = r.get("pnl_usd")
            # Persist trailing stop state
            for field in ("highest_price", "lowest_price", "trailing_stop_price"):
                if r.get(field) is not None:
                    entry[field] = r[field]
            entry["last_checked"] = datetime.now().isoformat()
            remaining_open.append(entry)

    save_open_positions(remaining_open)

    print(f"\n  Closed {len(closed_records)} position(s)  |  {len(remaining_open)} remain open")
    return closed_records


def _build_close_record(position: Dict, close_type: str, close_reason: str) -> Dict:
    """Build a close record for a position."""
    return {
        **position,
        "trade_result": "CLOSED_EARLY",
        "close_type": close_type,
        "close_reason": close_reason,
        "close_price": position.get("latest_price"),
        "resolved_at": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — NEW TRADE DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def _build_portfolio_state(open_positions: List[Dict], config: Config) -> str:
    """Build a portfolio state summary for injection into the LLM prompt."""
    if not open_positions:
        return (
            "PORTFOLIO STATE:\n"
            "  Positions: 0\n"
            "  Total invested: $0.00\n"
            "  Available slots: "
            f"{config.get('trading.max_open_positions', 20)}/{config.get('trading.max_open_positions', 20)}\n"
            "  No existing exposure."
        )

    total_invested = sum(p.get("amount_invested", 5) for p in open_positions)
    total_pnl_usd = sum(p.get("pnl_usd", 0) or 0 for p in open_positions)
    total_pnl_pct = (total_pnl_usd / total_invested * 100) if total_invested > 0 else 0
    max_open = config.get("trading.max_open_positions", 20)
    available_slots = max(0, max_open - len(open_positions))

    # Avg holding period
    hold_days = []
    for p in open_positions:
        exec_date = p.get("execution_date", "")
        if exec_date:
            try:
                opened = datetime.strptime(exec_date, "%Y-%m-%d")
                hold_days.append((datetime.now() - opened).days)
            except Exception:
                pass
    avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0

    lines = [
        "PORTFOLIO STATE:",
        f"  Positions: {len(open_positions)}/{max_open} (slots available: {available_slots})",
        f"  Total invested: ${total_invested:.2f}",
        f"  Unrealised P&L: ${total_pnl_usd:+.2f} ({total_pnl_pct:+.1f}%)",
        f"  Avg holding period: {avg_hold:.0f} days",
        "",
        "  Current holdings:",
    ]
    for p in open_positions:
        pnl_str = f"{p.get('pnl_pct', 0):+.1f}%" if p.get("pnl_pct") is not None else "N/A"
        lines.append(
            f"    - {p.get('symbol', '?').upper()} ({p.get('direction', 'LONG')}) "
            f"@ ${p.get('entry_price', '?')} | P&L: {pnl_str} | "
            f"Horizon: {p.get('time_horizon', '?')}"
        )
    return "\n".join(lines)


def _sanity_check_entry_price(
    trade: Dict, coin_data: Dict, max_deviation_pct: float = 2.0
) -> bool:
    """
    Verify Claude's suggested entry price is within max_deviation_pct of
    the actual CoinGecko price. Returns True if the price is sane.
    """
    llm_price = trade.get("entry_price")
    actual_price = coin_data.get("current_price")

    if llm_price is None or actual_price is None or actual_price == 0:
        return True  # can't check, let it through (will be overwritten anyway)

    deviation = abs(float(llm_price) - float(actual_price)) / float(actual_price) * 100
    if deviation > max_deviation_pct:
        logger.warning(
            f"Entry price sanity check failed for {trade.get('symbol', '?').upper()}: "
            f"LLM=${llm_price}, actual=${actual_price} (deviation: {deviation:.1f}%)"
        )
        return False
    return True


def run_phase2_new_trades(
    price_client: CryptoPriceClient,
    news_client: CryptoNewsClient,
    llm_analyzer: LLMAnalyzer,
    trade_executor: TradeExecutor,
    risk_manager: RiskManager,
    config: Config,
    prompts: dict,
    logger,
    profiles_data: dict = None,
) -> List[Dict]:
    """
    Phase 2: Discover and execute new crypto trades.

    Pipeline:
      1. Fetch top coins, trending, Fear & Greed
      2. Quantitative pre-screen: score coins on momentum, volume spike,
         relative strength vs BTC → top 10-15 candidates
      3. Volatility filter: reject coins where vol-adaptive stop fires within 1.5 daily moves
      4. Fetch news and filter to only articles mentioning shortlisted coins
      5. Build focused LLM prompt with portfolio state and lessons
      6. Validate LLM output: entry price sanity check, conviction filter
      7. Apply risk limits and record qualifying trades
    """
    print("\n" + "=" * 70)
    print("PHASE 2 — NEW TRADE DISCOVERY")
    print("=" * 70)

    open_positions = load_open_positions()
    held_coins = {p.get("coin_id", "") for p in open_positions}
    if held_coins:
        print(f"\n  Holding {len(held_coins)} open position(s) — those coins will be skipped")

    # --- Drawdown circuit breaker ---
    drawdown_status, drawdown_pct, drawdown_msg = risk_manager.check_portfolio_drawdown(open_positions)
    if drawdown_status == "halt":
        print(f"\n  DRAWDOWN CIRCUIT BREAKER TRIGGERED: {drawdown_msg}")
        print("  No new trades will be opened until portfolio recovers.")
        return []

    # --- Step 1: Fetch market data ---
    print("\n  [1/7] Fetching crypto market data...")

    coins = price_client.get_top_coins()
    if not coins:
        logger.error("Failed to fetch coin data")
        print("  ERROR: Could not fetch coins from CoinGecko")
        return []

    fear_greed = price_client.get_fear_greed_index()
    trending = price_client.get_trending_coins()
    fg_val = fear_greed.get("value", 50)
    print(f"  Coins: {len(coins)} | Fear & Greed: {fg_val} ({fear_greed.get('classification', '?')})")
    print(f"  Trending: {len(trending)} coins")
    if drawdown_status == "warn":
        print(f"  WARNING: {drawdown_msg}")

    # --- Regime-adaptive rules ---
    regime = get_market_regime(fg_val)
    print(f"\n  Market Regime: [{regime['name'].upper()}] — {regime['note']}")

    if regime["block_new_trades"]:
        print("  Phase 2 halted: market is in Extreme Greed — protecting capital.")
        return []

    # Use regime overrides for conviction and open-position cap
    regime_min_conviction = regime["min_conviction"]
    regime_max_open = regime["max_open_positions"]
    regime_rank_cap = regime["max_market_cap_rank"]  # None = unrestricted

    # --- Step 2: Quantitative pre-screen ---
    print("\n  [2/7] Running quantitative pre-screen...")
    min_volume = config.get("data_sources.min_volume_usd", 50_000_000)
    max_candidates = config.get("data_sources.max_candidates", 15)

    trending_ids = {t.get("id", "") for t in trending}
    screened = screen_coins(
        coins=coins,
        trending_ids=trending_ids,
        held_coin_ids=held_coins,
        min_volume_usd=min_volume,
        max_candidates=max_candidates,
        fear_greed_value=fg_val,
        max_market_cap_rank=regime_rank_cap,
        profiles_data=profiles_data,
    )

    if not screened:
        print("  No coins passed the pre-screen")
        return []

    # --- Step 3: Volatility filter ---
    print(f"\n  [3/7] Running volatility filter on {len(screened)} candidates...")
    min_stop_multiple = config.get("data_sources.min_stop_multiple", 1.5)
    screened, vol_rejected = enrich_with_volatility(
        screened, price_client,
        min_stop_multiple=min_stop_multiple,
    )
    if vol_rejected:
        vol_reject_names = [c.get("symbol", "?").upper() for c in vol_rejected]
        print(f"  Volatility rejected: {', '.join(vol_reject_names)}")
    print(f"  Volatility accepted: {len(screened)} coins")

    if not screened:
        print("  No coins passed the volatility filter")
        return []

    print(format_screen_summary(screened))

    # Build lookup of screened coin IDs and their data
    screened_lookup = {c.get("id", ""): c for c in screened}
    screened_symbols = [c.get("symbol", "").upper() for c in screened]
    screened_names = [c.get("name", "") for c in screened]

    # --- Step 4: Fetch and filter news ---
    print(f"\n  [4/7] Fetching crypto news (filtered to {len(screened)} shortlisted coins)...")
    all_news = news_client.get_all_news()
    filtered_news = news_client.filter_by_coins(all_news, screened_symbols, screened_names)
    print(f"  News: {len(filtered_news)} relevant articles (from {len(all_news)} total)")

    # --- Step 5: Build focused LLM prompt ---
    print("\n  [5/7] Analyzing with Claude...")

    crypto_data = price_client.format_for_llm(screened, fear_greed, trending)
    news_data = news_client.format_for_llm(filtered_news)
    portfolio_state = _build_portfolio_state(open_positions, config)

    # Stable strategy + rules + JSON schema → sent as cached system prompt
    system_prompt = prompts.get("crypto_analysis_system", "")

    # Dynamic per-run data → user message
    user_template = prompts.get("crypto_analysis_user", "")
    user_content = user_template.format(
        crypto_data=crypto_data,
        news_data=news_data,
        current_holdings=portfolio_state,
    )

    # Inject coin behavioral profiles (historical regime performance, vol, corr)
    screened_coin_ids = [c.get("id", "") for c in screened]
    profiles_text = format_profiles_for_llm(screened_coin_ids, profiles_data)
    if profiles_text:
        user_content += f"\n\n{profiles_text}"
    else:
        user_content += (
            "\n\n[No coin behavioral profiles available — "
            "run scripts/build_coin_profiles.py to generate them]"
        )

    # Inject lessons from previous sessions
    lessons_text = load_recent_lessons()
    if lessons_text:
        user_content += f"\n\n{lessons_text}"

    analysis = llm_analyzer.analyze(user_content, system=system_prompt)
    all_trades = analysis.get("trades", [])

    print(f"  Trades recommended: {len(all_trades)} | Sentiment: {analysis.get('overall_sentiment', 'N/A')}")
    print(f"  Market summary: {analysis.get('market_summary', 'N/A')[:100]}")

    # --- Step 6: Validate and enrich ---
    print(f"\n  [6/7] Validating recommendations...")

    if not all_trades:
        summary = analysis.get("market_summary", "")
        sentiment = analysis.get("overall_sentiment", "")
        msg = f"  No trades: {sentiment} — {summary[:120]}" if summary else "  No trading opportunities found this run"
        print(msg)
        logger.info(f"0-trade run | sentiment={sentiment} | {summary[:150]}")
        return []

    # Also keep original full coin lookup for any coin Claude might reference
    full_coin_lookup = {c.get("id", ""): c for c in coins}

    enriched: List[Dict] = []
    for trade in all_trades:
        coin_id = trade.get("coin_id", "")

        # Skip coins already held
        if coin_id in held_coins:
            logger.info(f"Skipping {coin_id} — already held")
            continue

        # Prefer screened data, fall back to full coin list
        coin_data = screened_lookup.get(coin_id) or full_coin_lookup.get(coin_id)
        if not coin_data:
            logger.warning(f"Coin {coin_id} not found in any lookup — skipping")
            trade_executor.save_rejected_trades([trade], f"Coin {coin_id} not in market data")
            continue

        # Entry price sanity check — reject if LLM hallucinated a price
        if not _sanity_check_entry_price(trade, coin_data):
            trade_executor.save_rejected_trades(
                [trade], f"Entry price sanity check failed (LLM=${trade.get('entry_price')}, "
                f"actual=${coin_data.get('current_price')})"
            )
            continue

        # Overwrite entry price with actual CoinGecko price
        trade["entry_price"] = coin_data.get("current_price")
        trade["market_cap"] = coin_data.get("market_cap")
        trade["volume_24h"] = coin_data.get("total_volume")
        trade["price_change_24h"] = coin_data.get("price_change_percentage_24h") or 0
        trade["direction"] = trade.get("direction", "LONG")

        # Carry over screen score and volatility data if available
        for field in ("screen_score", "daily_vol_pct", "stop_multiple", "vol_signal",
                      "adaptive_stop_pct"):
            if coin_data.get(field) is not None:
                trade[field] = coin_data[field]

        enriched.append(trade)

    # Filter by conviction — regime overrides config floor
    min_conviction = regime_min_conviction
    high_conviction = [t for t in enriched if float(t.get("conviction", 0)) >= min_conviction]
    low_conviction = [t for t in enriched if float(t.get("conviction", 0)) < min_conviction]
    if low_conviction:
        trade_executor.save_rejected_trades(low_conviction, f"Below min conviction ({min_conviction})")
        print(f"  Rejected {len(low_conviction)} trades below {min_conviction} conviction (regime: {regime['name']})")

    # Sector concentration check
    max_per_sector = config.get("trading.max_positions_per_sector", 4)
    high_conviction, sector_rejected = check_sector_concentration(
        high_conviction, open_positions, max_positions_per_sector=max_per_sector
    )
    if sector_rejected:
        syms = ", ".join(t.get("symbol", "?").upper() for t in sector_rejected)
        print(f"  Rejected {len(sector_rejected)} trades (sector concentration): {syms}")
        trade_executor.save_rejected_trades(
            sector_rejected,
            f"Sector concentration cap ({max_per_sector} per sector)"
        )

    # --- Step 7: Risk limits and execution ---
    print(f"\n  [7/7] Applying risk limits and recording trades...")

    # Cap by open position limit — regime may lower this below config default
    config_max_open = config.get("trading.max_open_positions", 20)
    max_open = min(config_max_open, regime_max_open)
    current_open = len(load_open_positions())
    open_slots = max(0, max_open - current_open)
    if open_slots == 0:
        print(f"  Open position cap reached ({current_open}/{max_open}) — no new trades (regime: {regime['name']})")
        trade_executor.save_rejected_trades(high_conviction, f"Open position cap reached ({max_open}, regime: {regime['name']})")
        return []

    # Cap at max trades per run
    max_per_run = config.get("trading.max_trades_per_run", 5)
    slots_remaining = min(open_slots, max_per_run)
    high_conviction.sort(key=lambda t: float(t.get("conviction", 0)), reverse=True)
    limited = high_conviction[:slots_remaining]
    overflow = high_conviction[slots_remaining:]
    if overflow:
        trade_executor.save_rejected_trades(overflow, "Trade cap reached (per-run limit)")
        print(f"  Selected top {len(limited)} trades by conviction ({len(overflow)} rejected — cap)")
    else:
        print(f"  {len(limited)} trade(s) selected")

    if not limited:
        print("  No trades after filters")
        return []

    # Attach investment amount
    max_pos_usd = config.get("trading.max_position_size_usd", 5.0)
    for trade in limited:
        trade["amount_invested"] = round(float(max_pos_usd), 2)

    # Global risk check
    today_key = datetime.now().strftime("%Y-%m-%d")
    today_pnl = PnLTracker().daily_summary().get(today_key, {}).get("pnl_usd", 0.0)
    allowed, reason = risk_manager.check_trade_limits(
        proposed_trades=limited,
        current_pnl=today_pnl,
        open_positions=open_positions,
        max_position_size_usd=max_pos_usd,
        trading_enabled=config.get("trading.enabled", False),
    )
    if not allowed:
        logger.warning(f"Risk check failed: {reason}")
        print(f"  {reason}")
        trade_executor.save_rejected_trades(limited, reason)
        return []

    # Execute (write to file)
    execution_results = trade_executor.execute_trades(limited)
    executed = execution_results["executed"]
    print(f"  Recorded {len(executed)} trade(s)")

    if execution_results["failed"]:
        print(f"  {len(execution_results['failed'])} trade(s) failed to record")

    # Add to open positions
    add_to_open_positions(executed)
    print(f"  Updated open positions file ({POSITIONS_FILE})")

    # Print trade summary
    if executed:
        print("\n  New trades:")
        for trade in executed:
            score_str = f" | Screen: {trade.get('screen_score', 'N/A')}" if trade.get("screen_score") is not None else ""
            vol_str = ""
            if trade.get("daily_vol_pct") is not None:
                vol_str = f" | Vol: {trade['daily_vol_pct']:.1f}%/d ({trade.get('stop_multiple', '?')}x)"
            print(
                f"    [{trade.get('direction', 'LONG')}] {trade.get('symbol', '?').upper()} "
                f"@ ${trade.get('entry_price', 0):,.2f}\n"
                f"      Conv: {float(trade.get('conviction', 0)):.2f} | "
                f"Size: ${trade.get('amount_invested', 5):.2f} | "
                f"Horizon: {trade.get('time_horizon', '?')}{score_str}{vol_str}\n"
                f"      {trade.get('reasoning', '')[:120]}..."
            )

    return executed


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — POST-TRADE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_phase3_post_trade_analysis(
    closed_records: List[Dict],
    llm_analyzer: LLMAnalyzer,
    config: Config,
    prompts: dict,
    logger,
) -> None:
    """
    Phase 3: Run LLM post-trade analysis on positions closed during Phase 1.
    Extract lessons and save to lessons.json for continuous learning.
    """
    print("\n" + "=" * 70)
    print("PHASE 3 — POST-TRADE ANALYSIS")
    print("=" * 70)

    if not closed_records:
        print("\n  No positions were closed this run — skipping analysis.")
        return

    print(f"\n  Analyzing {len(closed_records)} closed position(s)...")

    # Classify closed records by result for analysis
    for r in closed_records:
        if r.get("trade_result") == "CLOSED_EARLY":
            close_type = r.get("close_type", "")
            if close_type == "TAKE_PROFIT":
                r["trade_result"] = "WIN"
            elif close_type in ("CUT_LOSS", "STOP_LOSS"):
                r["trade_result"] = "LOSS"
            elif close_type in ("TRAILING_STOP", "TIME_EXPIRED"):
                # Classify by actual P&L
                r["trade_result"] = "WIN" if (r.get("pnl_pct") or 0) > 0 else "LOSS"
            else:
                r["trade_result"] = "LOSS"

    open_positions = load_open_positions()

    prompt_template = prompts.get("post_trade_analysis", "")
    trades_text = json.dumps(closed_records, indent=2, default=str)
    positions_text = json.dumps(open_positions, indent=2, default=str)
    prompt = prompt_template.format(
        trades=trades_text,
        positions=positions_text,
        date=datetime.now().strftime("%Y-%m-%d"),
    )

    # Post-trade analysis is summarization — use the cheaper lite model
    # (falls back to the default model if llm.lite_model is not configured)
    lite_model = config.get("llm.lite_model")
    analysis = llm_analyzer.analyze(prompt, model=lite_model)

    # Compute stats
    wins = [r for r in closed_records if r.get("trade_result") == "WIN"]
    win_rate = len(wins) / len(closed_records) * 100 if closed_records else 0
    total_pnl = sum(r.get("pnl_usd", 0) or 0 for r in closed_records)

    save_lessons({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "session": "intraday",
        "resolved_count": len(closed_records),
        "win_rate_pct": round(win_rate, 1),
        "pnl_usd": round(total_pnl, 2),
        **{k: v for k, v in analysis.items() if k not in ("date",)},
    })

    print(f"  Win rate: {win_rate:.0f}%  |  P&L: ${total_pnl:+.2f}")
    for lesson in analysis.get("lessons", [])[:3]:
        print(f"    - {lesson}")

    print(f"\n  Lessons saved to {LESSONS_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Crypto trading bot — intraday session")
    parser.add_argument("--skip-new-trades", action="store_true", help="Skip Phase 2 (new trade discovery)")
    parser.add_argument("--skip-closes", action="store_true", help="Skip Phase 1 (position management)")
    args = parser.parse_args()

    print("=" * 80)
    print("CRYPTO BOT — Intraday Trading Session")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # ── Setup ─────────────────────────────────────────────────────────────────
    config = Config()
    logger = setup_logger(
        log_level=config.get("logging.level", "INFO"),
        log_dir=config.get("logging.log_dir", "logs"),
        log_to_file=config.get("logging.log_to_file", True),
        script_name="intraday_trader",
    )

    # Load prompts
    with open(REPO_ROOT / "config" / "prompts.yaml") as f:
        prompts = yaml.safe_load(f)

    # ── Load coin behavioral profiles (built weekly by build_coin_profiles.py) ──
    profiles_data = load_profiles()

    # ── Shared components ─────────────────────────────────────────────────────
    price_client = CryptoPriceClient(
        top_n=config.get("data_sources.coingecko.top_coins", 50),
    )
    news_client = CryptoNewsClient(
        max_articles=config.get("data_sources.news.max_articles", 30),
    )
    llm_analyzer = LLMAnalyzer(
        api_key=config.get("llm.api_key"),
        model=config.get("llm.model"),
        max_tokens=config.get("llm.max_tokens", 4000),
        temperature=config.get("llm.temperature", 0.7),
    )
    trade_executor = TradeExecutor(enabled=config.get("trading.enabled", False))
    risk_manager = RiskManager(
        max_daily_loss_pct_of_committed=config.get("risk.max_daily_loss_pct_of_committed", 0.25),
        max_daily_loss_floor_usd=config.get("risk.max_daily_loss_floor_usd", 5.0),
        max_position_percentage=config.get("risk.max_position_percentage", 0.2),
        stop_loss_percentage=config.get("risk.stop_loss_percentage", 0.20),
        drawdown_warning_pct=config.get("risk.drawdown_warning_pct", -15.0),
        drawdown_halt_pct=config.get("risk.drawdown_halt_pct", -25.0),
    )

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    closed_records: List[Dict] = []
    if not args.skip_closes:
        closed_records = run_phase1_position_management(
            price_client=price_client,
            llm_analyzer=llm_analyzer,
            trade_executor=trade_executor,
            risk_manager=risk_manager,
            config=config,
            logger=logger,
        )

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    new_trades: List[Dict] = []
    if not args.skip_new_trades:
        new_trades = run_phase2_new_trades(
            price_client=price_client,
            news_client=news_client,
            llm_analyzer=llm_analyzer,
            trade_executor=trade_executor,
            risk_manager=risk_manager,
            config=config,
            prompts=prompts,
            logger=logger,
            profiles_data=profiles_data,
        )

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    run_phase3_post_trade_analysis(
        closed_records=closed_records,
        llm_analyzer=llm_analyzer,
        config=config,
        prompts=prompts,
        logger=logger,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SESSION SUMMARY")
    print("=" * 80)
    print(f"  Positions closed this run : {len(closed_records)}")
    print(f"  New trades executed       : {len(new_trades)}")
    open_count = len(load_open_positions())
    print(f"  Total open positions      : {open_count}")

    pnl_tracker = PnLTracker()
    all_time = pnl_tracker.all_time_summary()
    wr = f"{all_time['win_rate_pct']}%" if all_time["win_rate_pct"] is not None else "N/A"
    print(f"  All-time P&L              : ${all_time['total_pnl_usd']:+.2f}  (win rate: {wr})")

    print("\n" + "=" * 80)
    logger.info("Intraday trading session complete")
    print("Session complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
