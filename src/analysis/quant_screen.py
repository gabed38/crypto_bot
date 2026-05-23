"""
Quantitative pre-screen for crypto coins.

Scores and filters coins BEFORE sending to the LLM, reducing token usage
and improving analysis quality by focusing on the most interesting candidates.

Scoring factors:
  - Momentum: weighted combo of 1h, 24h, 7d price change
  - Volume spike: 24h volume relative to market cap (proxy for unusual activity)
  - Relative strength vs BTC: outperforming Bitcoin is a bullish signal
  - Trending bonus: coins on CoinGecko's trending list get a boost
  - Exhaustion penalty: coins already up >8% in Fear regimes get penalised (likely priced in)
  - Dip-buy bonus: coins down 5-25% with volume in Fear regimes get a bonus (mean-reversion)
  - Regime history bonus: coins with >65% win rate in the current F&G regime get a boost

Filters:
  - Minimum 24h volume floor (default $50M)
  - Exclude stablecoins
  - Exclude coins already held
"""

import math
import time
from typing import Dict, Any, List, Set, Optional, Tuple
from loguru import logger


# ── Sector mapping ────────────────────────────────────────────────────────────
# Broad sector labels for the top ~80 coins. Unknown coins fall back to "Other".
# Sectors: L1, L2, DeFi, Meme, AI, Gaming, Infra, Privacy, Exchange, BTC-Eco
COIN_SECTORS: Dict[str, str] = {
    # Layer 1
    "bitcoin": "L1", "ethereum": "L1", "solana": "L1", "avalanche-2": "L1",
    "cardano": "L1", "polkadot": "L1", "near": "L1", "aptos": "L1",
    "sui": "L1", "cosmos": "L1", "algorand": "L1", "tron": "L1",
    "hedera-hashgraph": "L1", "stellar": "L1", "internet-computer": "L1",
    "flow": "L1", "multiversx": "L1", "ton": "L1", "sei-network": "L1",
    "kaspa": "L1",
    # Layer 2 / scaling
    "matic-network": "L2", "polygon": "L2", "arbitrum": "L2",
    "optimism": "L2", "starknet": "L2", "zksync": "L2", "base": "L2",
    "loopring": "L2", "immutable-x": "L2", "mantle": "L2", "scroll": "L2",
    # DeFi
    "uniswap": "DeFi", "aave": "DeFi", "chainlink": "DeFi",
    "curve-dao-token": "DeFi", "maker": "DeFi", "compound-governance-token": "DeFi",
    "lido-dao": "DeFi", "synthetix-network-token": "DeFi", "1inch": "DeFi",
    "pancakeswap-token": "DeFi", "gmx": "DeFi", "jupiter-exchange-solana": "DeFi",
    "raydium": "DeFi", "pendle": "DeFi", "ethena": "DeFi",
    # Meme
    "dogecoin": "Meme", "shiba-inu": "Meme", "pepe": "Meme",
    "bonk": "Meme", "floki": "Meme", "dogwifcoin": "Meme",
    "brett": "Meme", "book-of-meme": "Meme",
    # AI / Data
    "render-token": "AI", "bittensor": "AI", "fetch-ai": "AI",
    "ocean-protocol": "AI", "akash-network": "AI", "artificial-superintelligence-alliance": "AI",
    "worldcoin-wld": "AI",
    # Gaming / Metaverse
    "the-sandbox": "Gaming", "decentraland": "Gaming", "axie-infinity": "Gaming",
    "gala": "Gaming", "illuvium": "Gaming", "pixels": "Gaming",
    # Infrastructure / Storage
    "filecoin": "Infra", "arweave": "Infra", "the-graph": "Infra",
    "helium": "Infra", "ankr": "Infra", "livepeer": "Infra",
    # Privacy
    "monero": "Privacy", "zcash": "Privacy", "dash": "Privacy",
    "oasis-network": "Privacy",
    # Exchange tokens
    "binancecoin": "Exchange", "okb": "Exchange", "crypto-com-chain": "Exchange",
    "gate-2": "Exchange", "kucoin-shares": "Exchange",
    # BTC ecosystem
    "wrapped-bitcoin": "BTC-Eco", "stacks": "BTC-Eco", "runes": "BTC-Eco",
}


def get_sector(coin_id: str) -> str:
    """Return the sector label for a coin, defaulting to 'Other'."""
    return COIN_SECTORS.get(coin_id, "Other")


def check_sector_concentration(
    proposed_trades: List[Dict[str, Any]],
    open_positions: List[Dict[str, Any]],
    max_positions_per_sector: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Flag proposed trades that would create excessive sector concentration.

    A trade is rejected if adding it would put the sector's total count
    (open + proposed-so-far) at or above max_positions_per_sector.

    Each coin in both lists should have a "sector" field; if absent,
    get_sector(coin_id) is used as a fallback.

    Args:
        proposed_trades:           Trades Claude wants to open, ordered by conviction desc.
        open_positions:            Currently open positions.
        max_positions_per_sector:  Hard cap per sector across open + new (default 4).

    Returns:
        (accepted, rejected) — rejected trades include a "sector_reject_reason" field.
    """
    # Count current sector exposure from open positions
    sector_counts: Dict[str, int] = {}
    for pos in open_positions:
        sector = pos.get("sector") or get_sector(pos.get("coin_id", ""))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    accepted = []
    rejected = []

    for trade in proposed_trades:
        coin_id = trade.get("coin_id", "")
        sector = trade.get("sector") or get_sector(coin_id)
        trade["sector"] = sector  # stamp it so it's persisted
        current_count = sector_counts.get(sector, 0)

        if current_count >= max_positions_per_sector:
            reason = (
                f"sector '{sector}' already has {current_count} positions "
                f"(cap: {max_positions_per_sector})"
            )
            trade["sector_reject_reason"] = reason
            logger.warning(
                f"Sector concentration reject {(trade.get('symbol') or coin_id).upper()}: {reason}"
            )
            rejected.append(trade)
        else:
            accepted.append(trade)
            sector_counts[sector] = current_count + 1  # count this new one

    return accepted, rejected


# ── Market regime ─────────────────────────────────────────────────────────────

def get_market_regime(fear_greed_value: int) -> Dict[str, Any]:
    """
    Translate the Fear & Greed Index into a trading regime with concrete rules.

    Conviction thresholds are derived empirically from 90-day historical data
    across 119 tradeable coins (data/coin_profiles/profiles.json, Apr 2026):

        Regime          Base win-rate   Avg 7d return   Conviction floor
        Extreme Fear    43.6 %          -0.3 %          0.70  ← BEST regime
        Fear            34.8 %          -2.2 %          0.77
        Neutral         22.0 %          -5.3 %          0.88  ← WORST regime
        Greed           no data*                        0.75
        Extreme Greed   no data*                        blocked

    * The 90-day window (Jan–Apr 2026) was a bear market with 59 Extreme Fear
      days and only 1 Greed day, so Greed/Extreme Greed thresholds remain
      conservative estimates. Refresh thresholds by re-running:
          python scripts/build_coin_profiles.py

    The counter-intuitive result: Extreme Fear is the easiest regime to trade
    (highest base win rate, market bounces from panic lows) so it carries the
    lowest conviction bar.  Neutral is the hardest — brief relief rallies tend
    to roll over quickly — so it carries the highest bar.

    Returns a dict with:
        name:                 Human-readable regime label
        min_conviction:       Minimum conviction score to open a trade
        max_open_positions:   Cap on total open positions in this regime
        max_market_cap_rank:  Only consider coins ranked at or above this rank
                              (None = no restriction)
        block_new_trades:     If True, skip Phase 2 entirely
        note:                 One-line explanation printed at runtime
    """
    fg = int(fear_greed_value)

    if fg <= 20:
        return {
            "name": "Extreme Fear",
            "min_conviction": 0.68,
            "max_open_positions": 15,
            "max_market_cap_rank": None,   # no rank cap — screener + vol filter + profiles handle quality
            "block_new_trades": False,
            "note": (
                "Extreme Fear — full universe, conviction ≥ 0.68, max 15 positions. "
                "Target dip bounces on mid/small caps (1–3d), avoid pumps >12% in 24h."
            ),
        }
    elif fg <= 40:
        return {
            "name": "Fear",
            "min_conviction": 0.68,
            "max_open_positions": 18,
            "max_market_cap_rank": None,   # no rank cap
            "block_new_trades": False,
            "note": (
                "Fear — full universe, conviction ≥ 0.68, max 18 positions. "
                "Dip entries and breakouts welcome. Avoid pumps >15% in 24h."
            ),
        }
    elif fg <= 60:
        return {
            "name": "Neutral",
            "min_conviction": 0.72,
            "max_open_positions": 20,
            "max_market_cap_rank": None,
            "block_new_trades": False,
            "note": (
                "Neutral — full universe, conviction ≥ 0.72, max 20 positions. "
                "Balanced dip and momentum plays."
            ),
        }
    elif fg <= 80:
        return {
            "name": "Greed",
            "min_conviction": 0.75,
            "max_open_positions": 20,
            "max_market_cap_rank": None,
            "block_new_trades": False,
            "note": (
                "Greed — conviction ≥ 0.75, max 20 positions. "
                "Favour breakouts and catalysts; avoid chasing extended rallies."
            ),
        }
    else:
        return {
            "name": "Extreme Greed",
            "min_conviction": 1.0,      # effectively unreachable
            "max_open_positions": 0,
            "max_market_cap_rank": None,
            "block_new_trades": True,
            "note": "Extreme Greed — market overheated, no new trades this run",
        }


# Stablecoins, wrapped tokens, and commodity-pegged tokens to always exclude
EXCLUDED_COINS = {
    # USD stablecoins
    "tether", "usd-coin", "dai", "binance-usd", "trueusd", "paxos-standard",
    "frax", "usdd", "first-digital-usd", "ethena-usde", "usual-usd",
    "usd1-wlfi", "usd1", "paypal-usd", "mountain-protocol-usdm",
    "ondo-us-dollar-yield", "ageur", "euro-coin", "usds", "sky-usds",
    "frax-usd", "crvusd", "usdf", "rlusd", "bfusd", "usdy", "cusd", "jusd",
    "pusd", "deusd", "fdusd", "gho", "ausd", "zusd", "nusd", "susd",
    # Yield-bearing / RWA stablecoins (0% vol, no alpha)
    "ondo-us-dollar-yield", "mountain-protocol-usdm", "usual-usd",
    "jtrsy", "ustb", "ousg", "usyc", "jaaa", "prime",
    # Non-crypto financial instruments leaking into CoinGecko
    "figure-heloc", "figure-heloc-2", "figure-heloc-3",  # FIGR_HELOC family
    "backed-ib01", "backed-ibtc", "backed-buidl",
    # Wrapped / liquid staking (track underlying, not tradeable alpha)
    "wrapped-bitcoin", "wrapped-ether", "staked-ether", "wrapped-steth",
    "coinbase-wrapped-btc", "rocket-pool-eth", "wrapped-eeth",
    "reth", "cbeth", "oseth", "meth", "sfrxeth", "weeth",
    # Gold / commodity-pegged (track gold, not crypto)
    "pax-gold", "tether-gold", "paxg", "xaut",
    # Exchange-specific tokens that are illiquid outside their native exchange
    "bfusd",
}


def _fg_to_regime_key(fg: int) -> str:
    """Map Fear & Greed value to coin profile regime key."""
    if fg <= 20:
        return "extreme_fear"
    elif fg <= 40:
        return "fear"
    elif fg <= 60:
        return "neutral"
    elif fg <= 80:
        return "greed"
    return "extreme_greed"


def screen_coins(
    coins: List[Dict[str, Any]],
    btc_data: Optional[Dict[str, Any]] = None,
    trending_ids: Optional[Set[str]] = None,
    held_coin_ids: Optional[Set[str]] = None,
    min_volume_usd: float = 50_000_000,
    max_candidates: int = 15,
    fear_greed_value: int = 50,
    max_market_cap_rank: Optional[int] = None,
    profiles_data: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Score and rank coins, returning the top candidates for LLM analysis.

    Each coin gets a composite score based on momentum, volume activity,
    and relative strength. Stablecoins, low-volume coins, and already-held
    coins are filtered out before scoring.

    Args:
        coins: Raw coin data from CoinGecko /coins/markets
        btc_data: Bitcoin's data dict (for relative strength calc). If None,
                  extracted from coins list automatically.
        trending_ids: Set of coin IDs currently trending on CoinGecko
        held_coin_ids: Set of coin IDs already in open positions
        min_volume_usd: Minimum 24h volume to consider (default $50M)
        max_candidates: Maximum coins to return
        fear_greed_value: Current Fear & Greed Index (0-100), used to adjust
                         how aggressive the screen is
        max_market_cap_rank: If set, only consider coins ranked at or above
                             this rank by market cap (e.g. 10 → top-10 only).
                             None means no restriction.
        profiles_data: Optional coin behavioral profiles dict (from
                       load_profiles()). If supplied, adds a regime performance
                       bonus/penalty based on each coin's historical win rate
                       in the current Fear & Greed regime.

    Returns:
        List of coin dicts, sorted by score descending, enriched with
        'screen_score' and 'screen_reasons' fields
    """
    if not coins:
        return []

    trending_ids = trending_ids or set()
    held_coin_ids = held_coin_ids or set()

    # Extract BTC data for relative strength calculation
    if btc_data is None:
        for c in coins:
            if c.get("id") == "bitcoin":
                btc_data = c
                break
    btc_24h = (btc_data.get("price_change_percentage_24h") or 0) if btc_data else 0
    btc_7d = (btc_data.get("price_change_percentage_7d_in_currency") or 0) if btc_data else 0

    scored = []
    filtered_reasons = {}

    for coin in coins:
        coin_id = coin.get("id", "")
        symbol = (coin.get("symbol") or "").upper()

        # --- Hard filters ---
        if coin_id in EXCLUDED_COINS:
            continue
        if coin_id in held_coin_ids:
            filtered_reasons[coin_id] = "already held"
            continue

        # Symbol-based stablecoin/peg filter for coins not in the explicit exclusion list
        sym_lower = symbol.lower()
        if any(sym_lower.endswith(s) for s in ("usd", "usdt", "usdc", "dai", "eur", "gbp")):
            filtered_reasons[coin_id] = f"stablecoin symbol pattern ({symbol})"
            continue

        # Hard filter: exhaustion pumps — thresholds vary by regime.
        # Coins already up big have priced in their catalyst; entering is chasing.
        # Thresholds:
        #   Extreme Fear (≤20):  >12% — panic-market pumps reverse fast
        #   Fear        (21-40): >18% — slightly more room for sustained moves
        #   Neutral+:            >30% — allow breakout plays; only filter extreme parabolic runs
        change_24h_raw = float(coin.get("price_change_percentage_24h") or 0)
        if fear_greed_value <= 20:
            exhaustion_threshold = 12
        elif fear_greed_value <= 40:
            exhaustion_threshold = 18
        else:
            exhaustion_threshold = 30   # only catch extreme parabolic runs in neutral/greed

        if change_24h_raw > exhaustion_threshold:
            filtered_reasons[coin_id] = (
                f"exhaustion hard-filter: +{change_24h_raw:.1f}% in 24h "
                f"(threshold {exhaustion_threshold}% for {_fg_to_regime_key(fear_greed_value)})"
            )
            continue

        # Regime-based market cap rank restriction
        if max_market_cap_rank is not None:
            rank = coin.get("market_cap_rank") or 9999
            if rank > max_market_cap_rank:
                filtered_reasons[coin_id] = (
                    f"rank {rank} > regime cap {max_market_cap_rank}"
                )
                continue

        volume = float(coin.get("total_volume") or 0)
        if volume < min_volume_usd:
            filtered_reasons[coin_id] = f"volume ${volume / 1e6:.0f}M < ${min_volume_usd / 1e6:.0f}M floor"
            continue

        # --- Scoring ---
        score = 0.0
        reasons = []

        # 1. Momentum score (weighted: 1h=0.2, 24h=0.5, 7d=0.3)
        change_1h = float(coin.get("price_change_percentage_1h_in_currency") or 0)
        change_24h = float(coin.get("price_change_percentage_24h") or 0)
        change_7d = float(coin.get("price_change_percentage_7d_in_currency") or 0)

        momentum = (change_1h * 0.2) + (change_24h * 0.5) + (change_7d * 0.3)
        # Normalize: +10% momentum → +5 score points, cap at +-10
        momentum_score = max(-10, min(10, momentum * 0.5))
        score += momentum_score
        if momentum > 3:
            reasons.append(f"strong momentum ({momentum:+.1f})")
        elif momentum < -3:
            reasons.append(f"weak momentum ({momentum:+.1f})")

        # 2. Volume spike (volume / market_cap ratio, higher = more unusual activity)
        market_cap = float(coin.get("market_cap") or 1)
        vol_ratio = volume / market_cap if market_cap > 0 else 0
        # Typical vol/mcap is ~0.03-0.10. Above 0.15 is notable.
        vol_score = min(5, max(0, (vol_ratio - 0.05) * 40))
        score += vol_score
        if vol_ratio > 0.15:
            reasons.append(f"volume spike (vol/mcap={vol_ratio:.2f})")

        # 3. Relative strength vs BTC
        rs_24h = change_24h - btc_24h
        rs_7d = change_7d - btc_7d
        rs_combined = (rs_24h * 0.6) + (rs_7d * 0.4)
        rs_score = max(-5, min(5, rs_combined * 0.3))
        score += rs_score
        if rs_combined > 3:
            reasons.append(f"outperforming BTC ({rs_combined:+.1f}%)")

        # 4. Trending bonus
        if coin_id in trending_ids:
            score += 3.0
            reasons.append("trending on CoinGecko")

        # 5. Volatility-capture signals
        # 5a. Soft exhaustion penalty (applies even when hard filter didn't remove the coin)
        if fear_greed_value <= 20 and change_24h > 8:
            penalty = min(6.0, (change_24h - 8) * 0.8)
            score -= penalty
            reasons.append(f"exhaustion penalty ({change_24h:+.0f}% in 24h, likely priced in)")
        elif fear_greed_value <= 40 and change_24h > 12:
            penalty = min(5.0, (change_24h - 12) * 0.6)
            score -= penalty
            reasons.append(f"exhaustion penalty ({change_24h:+.0f}% in 24h)")

        # 5b. Dip-buy opportunity: quality pullback with volume → mean-reversion bounce
        # Active across all regimes — volatile swings happen regardless of F&G
        if -30 <= change_24h <= -4 and vol_ratio > 0.01:
            dip_score = min(6.0, abs(change_24h) * 0.3)
            score += dip_score
            reasons.append(f"dip setup ({change_24h:+.1f}% pullback, volume present)")

        # 6. Historical regime performance bonus (uses coin behavioral profiles)
        if profiles_data is not None:
            raw_profiles = profiles_data.get("profiles", profiles_data)
            coin_profile = raw_profiles.get(coin_id, {})
            regime_key = _fg_to_regime_key(fear_greed_value)
            regime_stats = coin_profile.get("regime_returns", {}).get(regime_key, {})
            n = regime_stats.get("n", 0)
            if n >= 5:
                win_rate = regime_stats.get("win_rate", 0.5)
                avg_7d = regime_stats.get("avg_7d_pct", 0)
                if win_rate >= 0.65 and avg_7d >= 2.0:
                    score += 2.0
                    reasons.append(
                        f"strong regime history ({win_rate * 100:.0f}% wr, "
                        f"+{avg_7d:.1f}% avg in {regime_key})"
                    )
                elif win_rate >= 0.55 and avg_7d >= 0:
                    score += 0.75
                    reasons.append(
                        f"good regime history ({win_rate * 100:.0f}% wr in {regime_key})"
                    )
                elif win_rate <= 0.30:
                    score -= 1.5
                    reasons.append(
                        f"poor regime history ({win_rate * 100:.0f}% wr in {regime_key})"
                    )

        # 7. Fear & Greed rank adjustment
        # In extreme fear, prefer large caps (rank <= 10); in extreme greed, penalize less
        rank = coin.get("market_cap_rank") or 50
        if fear_greed_value < 25 and rank > 20:
            score -= 2.0
            reasons.append("penalized (small cap in fear market)")
        elif fear_greed_value > 75 and rank > 30:
            score -= 1.0
            reasons.append("penalized (small cap in greedy market)")

        coin_scored = {
            **coin,
            "screen_score": round(score, 2),
            "screen_reasons": reasons,
            "momentum": round(momentum, 2),
            "vol_ratio": round(vol_ratio, 4),
            "rs_vs_btc": round(rs_combined, 2),
            "sector": get_sector(coin_id),
        }
        scored.append(coin_scored)

    # Sort by score descending
    scored.sort(key=lambda c: c["screen_score"], reverse=True)

    # Log filtering stats
    n_filtered = len(filtered_reasons)
    if n_filtered:
        logger.info(f"Pre-screen filtered out {n_filtered} coins (held/low-volume/stablecoin)")
    logger.info(
        f"Pre-screen scored {len(scored)} coins, "
        f"returning top {min(max_candidates, len(scored))}"
    )

    return scored[:max_candidates]


def compute_volatility_stats(
    price_history: List[float],
    stop_loss_pct: float = 20.0,
) -> Dict[str, Any]:
    """
    Compute daily volatility stats from a list of closing prices.

    Uses log returns over the provided history to estimate typical daily
    price movement.  ``stop_loss_pct`` is used only to compute the initial
    ``stop_multiple``; callers that use vol-adaptive stops should overwrite
    ``stop_multiple`` after receiving this result.

    Args:
        price_history: List of daily closing prices, oldest first.
        stop_loss_pct: Reference stop-loss percentage for stop_multiple calc.

    Returns dict with:
        daily_vol_pct:  Average daily price swing as a percentage.
        stop_multiple:  How many daily moves the stop-loss represents.
                        Low values (< 2) mean the stop fires on normal noise.
        vol_signal:     "LOW" | "MEDIUM" | "HIGH" | "EXTREME" | "UNKNOWN"
    """
    if len(price_history) < 3:
        return {"daily_vol_pct": None, "stop_multiple": None, "vol_signal": "UNKNOWN"}

    returns = [
        math.log(price_history[i] / price_history[i - 1])
        for i in range(1, len(price_history))
        if price_history[i - 1] > 0 and price_history[i] > 0
    ]

    if not returns:
        return {"daily_vol_pct": None, "stop_multiple": None, "vol_signal": "UNKNOWN"}

    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / max(n - 1, 1)
    daily_vol = math.sqrt(variance) * 100  # convert to percentage

    stop_multiple = stop_loss_pct / daily_vol if daily_vol > 0 else float("inf")

    if daily_vol < 2.0:
        vol_signal = "LOW"
    elif daily_vol < 5.0:
        vol_signal = "MEDIUM"
    elif daily_vol < 10.0:
        vol_signal = "HIGH"
    else:
        vol_signal = "EXTREME"

    return {
        "daily_vol_pct": round(daily_vol, 2),
        "stop_multiple": round(stop_multiple, 2),
        "vol_signal": vol_signal,
    }


def enrich_with_volatility(
    screened: List[Dict[str, Any]],
    price_client,  # CryptoPriceClient — not typed to avoid circular import
    stop_loss_pct: float = 20.0,
    min_stop_multiple: float = 1.5,
    api_sleep_sec: float = 1.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch 14-day price history for each screened coin and compute volatility.

    Vol-adaptive stop: each coin's stop-loss is computed as 3× its daily
    volatility, capped at 25 % and floored at 10 %:

        adaptive_stop_pct = max(min(daily_vol_pct × 3, 25.0), 10.0)

    Examples:
        BTC/ETH tier   2 %/day  → 10 % stop (floor)
        Typical alt    4 %/day  → 12 % stop
        High-vol alt   7 %/day  → 21 % stop
        Extreme alt   10 %/day  → 25 % stop (cap)

    This gives low-vol coins a tighter stop (they move less, so 20 % flat was
    9× daily noise) while letting high-vol coins breathe without widening past
    25 %.

    Coins are hard-rejected if ``adaptive_stop_pct / daily_vol_pct`` (the
    stop_multiple) falls below ``min_stop_multiple`` — this only triggers for
    extreme-vol coins (>16 %/day) where the 25 % cap means the stop is less
    than 1.5× daily noise.

    A sleep between API calls respects CoinGecko's free-tier rate limit (~30/min).

    Args:
        screened:          Output of screen_coins() — top N candidates.
        price_client:      CryptoPriceClient instance.
        stop_loss_pct:     Unused — kept for backward compat; adaptive stop is
                           computed per-coin from daily_vol_pct.
        min_stop_multiple: Minimum acceptable stop/vol ratio (default 1.5).
        api_sleep_sec:     Sleep between CoinGecko calls (default 1.5s).

    Returns:
        (accepted, rejected) — two lists of coin dicts enriched with vol fields.
        Accepted coins have ``adaptive_stop_pct`` stamped on them.
        Rejected coins include a ``vol_reject_reason`` field.
    """
    accepted = []
    rejected = []

    # Brief pause before starting: the earlier market-data fetches (coins, trending,
    # fear & greed) consume part of CoinGecko's burst window.  Waiting a few seconds
    # lets the sliding window partially reset so the price-history calls don't 429.
    time.sleep(5)

    for i, coin in enumerate(screened):
        coin_id = coin.get("id", "")
        symbol = (coin.get("symbol") or "?").upper()

        history = price_client.get_price_history(coin_id, days=14)
        # After the 4th call CoinGecko's burst window resets — use a longer sleep
        sleep = api_sleep_sec if i < 4 else api_sleep_sec * 2
        time.sleep(sleep)

        if not history:
            logger.warning(f"No price history for {symbol} — keeping with UNKNOWN volatility")
            coin.update({"daily_vol_pct": None, "stop_multiple": None, "vol_signal": "UNKNOWN"})
        else:
            vol_stats = compute_volatility_stats(history)
            coin.update(vol_stats)

            daily_vol = vol_stats.get("daily_vol_pct", 0) or 0

            # Compute vol-adaptive stop: 3× daily vol, capped at 25%, floored at 10%
            if daily_vol > 0:
                adaptive_stop = round(max(min(daily_vol * 3.0, 25.0), 10.0), 1)
                adaptive_mult = round(adaptive_stop / daily_vol, 2)
            else:
                adaptive_stop = 20.0  # fallback if vol unknown
                adaptive_mult = vol_stats.get("stop_multiple")

            coin["adaptive_stop_pct"] = adaptive_stop
            coin["stop_multiple"] = adaptive_mult  # overwrite with adaptive ratio

            stop_mult = adaptive_mult

            if stop_mult is not None and stop_mult < min_stop_multiple:
                reason = (
                    f"stop_multiple={stop_mult:.2f} < {min_stop_multiple} "
                    f"(daily_vol={daily_vol:.1f}%, adaptive_stop={adaptive_stop}%, "
                    f"stop-loss fires on noise)"
                )
                coin["vol_reject_reason"] = reason
                logger.warning(f"Volatility reject {symbol}: {reason}")
                rejected.append(coin)
                continue

            mult_str = f"{stop_mult:.2f}x" if stop_mult is not None else "unknown"
            logger.info(
                f"Vol OK {symbol}: daily_vol={daily_vol:.1f}%  "
                f"adaptive_stop={adaptive_stop:.1f}%  "
                f"stop={mult_str} daily moves  signal={vol_stats.get('vol_signal')}"
            )

        # ── Intraday RSI (1h chart) ──────────────────────────────────────────
        # Fetch after the vol history call to stay within rate-limit budget.
        # CoinGecko free tier: ~30 req/min.  We sleep api_sleep_sec between
        # calls, so the combined vol + RSI calls per coin consume ≈3s.
        time.sleep(api_sleep_sec)
        intraday = price_client.get_intraday_rsi(coin_id)
        rsi = intraday.get("rsi")
        recent_4h = intraday.get("recent_4h_pct")
        coin["intraday_rsi"] = rsi
        coin["recent_4h_pct"] = recent_4h

        change_24h = float(coin.get("price_change_percentage_24h") or 0)

        # Hard filter: dip-buy candidate whose RSI has already recovered past 55
        # means the bounce is largely done — not a fresh entry point.
        # (RSI filter only blocks dip candidates; momentum/breakout coins are fine.)
        if change_24h <= -4 and rsi is not None and rsi > 55:
            reason = (
                f"dip-recovery filter: {change_24h:+.1f}% in 24h "
                f"but RSI={rsi:.0f} > 55 (bounce already underway, entry too late)"
            )
            coin["vol_reject_reason"] = reason
            logger.warning(f"RSI recovery reject {symbol}: {reason}")
            rejected.append(coin)
            continue

        # RSI scoring: adjust screen_score now that we have intraday data.
        # This re-orders the final candidate list so the most oversold coins
        # rank highest regardless of their 24h/7d momentum label.
        if rsi is not None:
            if rsi < 30:
                coin["screen_score"] = round(coin.get("screen_score", 0) + 4.0, 2)
                coin.setdefault("screen_reasons", []).append(f"deeply oversold RSI={rsi:.0f}")
                logger.info(f"RSI bonus {symbol}: RSI={rsi:.0f} → +4.0 (deeply oversold)")
            elif rsi < 40:
                coin["screen_score"] = round(coin.get("screen_score", 0) + 2.0, 2)
                coin.setdefault("screen_reasons", []).append(f"oversold RSI={rsi:.0f}")
                logger.info(f"RSI bonus {symbol}: RSI={rsi:.0f} → +2.0 (oversold)")
            elif rsi > 65:
                coin["screen_score"] = round(coin.get("screen_score", 0) - 2.0, 2)
                coin.setdefault("screen_reasons", []).append(f"overbought RSI={rsi:.0f}")
                logger.info(f"RSI penalty {symbol}: RSI={rsi:.0f} → -2.0 (overbought)")

        accepted.append(coin)

    logger.info(
        f"Volatility enrichment: {len(accepted)} accepted, "
        f"{len(rejected)} rejected (stop fires on noise)"
    )
    return accepted, rejected


def format_screen_summary(screened: List[Dict]) -> str:
    """Format a brief summary of the screen results for logging (includes vol + RSI if available)."""
    if not screened:
        return "No coins passed the pre-screen."
    lines = ["Pre-screen results:"]
    for i, c in enumerate(screened, 1):
        symbol = (c.get("symbol") or "?").upper()
        score = c.get("screen_score", 0)
        reasons = ", ".join(c.get("screen_reasons", [])) or "baseline"
        vol_str = ""
        if c.get("daily_vol_pct") is not None:
            adaptive = c.get("adaptive_stop_pct", "?")
            adaptive_str = f"{adaptive:.0f}%" if isinstance(adaptive, float) else adaptive
            vol_str = (
                f"  vol={c['daily_vol_pct']:.1f}%/day"
                f"  stop={adaptive_str}"
                f"  ({c.get('stop_multiple', '?')}x)"
            )
        rsi = c.get("intraday_rsi")
        rsi_str = f"  RSI={rsi:.0f}" if rsi is not None else ""
        r4h = c.get("recent_4h_pct")
        r4h_str = f"  4h={r4h:+.1f}%" if r4h is not None else ""
        lines.append(f"  {i:>2}. {symbol:<8} score={score:>+6.2f}{vol_str}{rsi_str}{r4h_str}  ({reasons})")
    return "\n".join(lines)
