"""
Coin Profiles — loader and LLM formatter.

Profiles are built by:  scripts/build_coin_profiles.py
Stored at:              data/coin_profiles/profiles.json

This module is used by intraday_trader.py to inject a compact behavioral
summary for every coin in the current screening session, giving the LLM
concrete historical context:

  - Typical daily volatility (how noisy is this coin?)
  - BTC correlation and beta (does it amplify BTC moves?)
  - Momentum persistence (does trend continue, or mean-revert?)
  - Worst / average drawdown over the profile window
  - Regime-conditioned 7-day forward returns (how did it historically
    perform when the Fear & Greed Index was in each band?)

Usage:
    from src.analysis.coin_profiles import load_profiles, format_profiles_for_llm

    profiles_data = load_profiles()
    text = format_profiles_for_llm(screened_coin_ids, profiles_data)
    prompt += f"\n\n{text}"
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

REPO_ROOT = Path(__file__).parent.parent.parent
PROFILES_FILE = REPO_ROOT / "data" / "coin_profiles" / "profiles.json"

# Short labels for regime buckets, ordered from most fearful to most greedy
_REGIME_ORDER = [
    ("extreme_fear", "ExtFear"),
    ("fear", "Fear"),
    ("neutral", "Neutral"),
    ("greed", "Greed"),
    ("extreme_greed", "ExtGreed"),
]


def load_profiles() -> Dict[str, Any]:
    """
    Load coin profiles from disk.

    Returns the full data dict (keys: updated_at, history_days, coin_count,
    profiles), or an empty dict if the file is missing or unreadable.

    Call this once at startup and pass the result to format_profiles_for_llm()
    to avoid repeated disk reads.
    """
    if not PROFILES_FILE.exists():
        logger.debug(
            "Coin profiles file not found — "
            "run scripts/build_coin_profiles.py to generate it"
        )
        return {}
    try:
        with open(PROFILES_FILE) as f:
            data = json.load(f)
        n = data.get("coin_count", 0)
        updated = data.get("updated_at", "?")[:10]
        logger.info(f"Loaded coin profiles: {n} coins (updated {updated})")
        return data
    except Exception as e:
        logger.warning(f"Failed to load coin profiles: {e}")
        return {}


def format_profiles_for_llm(
    coin_ids: List[str],
    profiles_data: Optional[Dict[str, Any]] = None,
    max_coins: int = 20,
) -> str:
    """
    Format behavioral profiles for the coins in this screening session.

    Only includes coins for which a profile exists.  The output is a compact
    multi-line text block suitable for appending to the LLM analysis prompt.

    Args:
        coin_ids:      CoinGecko IDs for the coins in this session (screened list).
        profiles_data: Pre-loaded profiles dict.  If None, loads from disk.
        max_coins:     Cap the number of profiles to keep the prompt concise.

    Returns:
        Formatted text block, or "" if no profiles are available.
    """
    if profiles_data is None:
        profiles_data = load_profiles()

    if not profiles_data:
        return ""

    all_profiles = profiles_data.get("profiles", {})
    updated_at = profiles_data.get("updated_at", "?")[:10]
    history_days = profiles_data.get("history_days", 90)

    # Match coin_ids that have profiles, preserve screening order
    matched = [(cid, all_profiles[cid]) for cid in coin_ids if cid in all_profiles]
    matched = matched[:max_coins]

    if not matched:
        return ""

    lines = [
        f"COIN BEHAVIOR PROFILES ({history_days}d historical data, "
        f"updated {updated_at}):",
        "Each row: volatility per day | BTC correlation | beta | "
        "momentum character | worst drawdown over period",
        "Regime row: avg 7-day forward return | win% (n=observations) "
        "for each Fear & Greed band when this coin was entered",
        "",
    ]

    for coin_id, p in matched:
        symbol = p.get("symbol", coin_id.upper())
        vol = p.get("daily_vol_pct", 0)
        corr = p.get("btc_corr", 0)
        beta = p.get("btc_beta", 1.0)
        persist = p.get("momentum_persistence", 0)
        worst_dd = p.get("worst_drawdown_pct", 0)
        avg_dd = p.get("avg_drawdown_pct", 0)

        # Human-readable momentum label
        if persist > 0.15:
            persist_label = "trending"
        elif persist > 0.05:
            persist_label = "slight trend"
        elif persist < -0.15:
            persist_label = "mean-reverting"
        elif persist < -0.05:
            persist_label = "slight reversion"
        else:
            persist_label = "random-walk"

        lines.append(
            f"[{symbol}] vol={vol:.1f}%/d | corr={corr:+.2f} | "
            f"beta={beta:.1f}x | {persist_label} ({persist:+.2f}) | "
            f"worst-dd={worst_dd:.0f}%  avg-dd={avg_dd:.1f}%"
        )

        # Regime returns: one line, skip buckets with n=0
        rr = p.get("regime_returns", {})
        regime_parts = []
        for key, label in _REGIME_ORDER:
            r = rr.get(key, {})
            n = r.get("n", 0)
            if n == 0:
                continue
            avg_7d = r.get("avg_7d_pct")
            wr = r.get("win_rate")
            if avg_7d is None or wr is None:
                continue
            sign = "+" if avg_7d >= 0 else ""
            regime_parts.append(
                f"{label}(n={n}):{sign}{avg_7d:.0f}%|{int(wr * 100)}%win"
            )

        if regime_parts:
            lines.append("  Regimes → " + "  ".join(regime_parts))

        lines.append("")  # blank line between coins

    # Strip trailing blank line
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def get_profile(coin_id: str, profiles_data: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
    """
    Return the raw profile dict for a single coin, or None if not found.

    Useful for checking a specific coin's historical stats in code.
    """
    if profiles_data is None:
        profiles_data = load_profiles()
    return profiles_data.get("profiles", {}).get(coin_id)
