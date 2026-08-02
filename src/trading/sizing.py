"""
Volatility-targeted position sizing.

The bot was staking a flat $5 on every trade regardless of how violently the coin
moves, which means a 2%/day coin and a 15%/day coin carried wildly different risk
for identical notional. Sizing inversely to volatility equalises that: each
position contributes roughly the same expected daily P&L swing.

This is the single best-supported idea in the momentum literature — volatility
management of momentum strategies has been found to roughly double Sharpe and
largely remove the crash tail, which matters here because crypto momentum is
specifically documented as crash-prone.

The input this needs (`daily_vol_pct`) is already computed for every candidate by
the volatility filter in the quant screen; it was simply being discarded.
"""

from typing import Dict, Optional

# A position sized to this much expected daily move is the unit of risk.
# At 3% target and a 6%/day coin, you take a half-size position.
DEFAULT_TARGET_VOL_PCT = 3.0


def vol_target_size(
    base_usd: float,
    daily_vol_pct: Optional[float],
    target_vol_pct: float = DEFAULT_TARGET_VOL_PCT,
    min_usd: float = 1.0,
    max_usd: Optional[float] = None,
) -> float:
    """Scale a base stake so each position carries comparable risk.

    size = base * (target_vol / daily_vol), clamped.

    When volatility is unknown the base size is returned unchanged — the screen
    marks those candidates UNKNOWN and they should not be silently up-sized on
    the basis of missing data.
    """
    try:
        base = float(base_usd)
    except (TypeError, ValueError):
        return 0.0

    if daily_vol_pct is None:
        return round(base, 2)
    try:
        vol = float(daily_vol_pct)
    except (TypeError, ValueError):
        return round(base, 2)
    if vol <= 0:
        return round(base, 2)

    scaled = base * (float(target_vol_pct) / vol)
    if max_usd is not None:
        scaled = min(scaled, float(max_usd))
    scaled = max(scaled, float(min_usd))
    return round(scaled, 2)


def size_crypto_trade(trade: Dict, config: Dict) -> Dict:
    """Compute sizing fields for one crypto trade, returned as a dict to merge.

    Keeps the inputs alongside the output so later analysis can separate a bad
    entry from a badly sized one.
    """
    method = config.get("sizing_method", "flat")
    base = config.get("base_position_usd", 5.0)
    max_usd = config.get("max_position_size_usd", 15.0)
    min_usd = config.get("min_position_size_usd", 1.0)
    target_vol = config.get("target_vol_pct", DEFAULT_TARGET_VOL_PCT)

    if method != "vol_target":
        return {
            "amount_invested": round(float(base), 2),
            "sizing_method": "flat",
        }

    vol = trade.get("daily_vol_pct")
    amount = vol_target_size(
        base_usd=base,
        daily_vol_pct=vol,
        target_vol_pct=target_vol,
        min_usd=min_usd,
        max_usd=max_usd,
    )
    return {
        "amount_invested": amount,
        "sizing_method": "vol_target",
        "target_vol_pct": target_vol,
        "sizing_daily_vol_pct": vol,
        "sizing_scale_factor": round(amount / base, 3) if base else None,
    }
