"""
Triple-barrier labeling.

The bot already closes positions three ways — target hit, stop hit, time elapsed —
but treats them as ad-hoc rules rather than as a labeling scheme. Formalising them
as barriers does two things:

  1. Scales the barriers to each coin's volatility instead of using fixed
     percentages, so a 2%/day coin and a 12%/day coin get barriers that are
     equally reachable rather than equally wide.
  2. Produces a clean training label (+1 / -1 / 0) for every closed position,
     which is what the secondary meta-labeling model will eventually need. Without
     this the trade history isn't usable for supervised learning.

Labels record which barrier was touched first, not simply whether the trade made
money — a position that drifted to +0.3% before expiring is a different event from
one that ran to its target, even though both are "wins" by P&L sign.
"""

from typing import Dict, Optional

# Barrier widths in units of daily volatility.
DEFAULT_PROFIT_MULT = 2.0
DEFAULT_STOP_MULT = 2.0


def compute_barriers(
    daily_vol_pct: Optional[float],
    profit_mult: float = DEFAULT_PROFIT_MULT,
    stop_mult: float = DEFAULT_STOP_MULT,
    fallback_profit_pct: float = 6.0,
    fallback_stop_pct: float = 10.0,
    min_profit_pct: float = 3.0,
    max_profit_pct: float = 25.0,
    min_stop_pct: float = 5.0,
    max_stop_pct: float = 25.0,
) -> Dict[str, float]:
    """Volatility-scaled take-profit and stop-loss barriers.

    Returns percentages, both positive. Falls back to fixed widths when the
    coin's volatility is unknown rather than guessing.
    """
    if daily_vol_pct is None:
        return {
            "profit_barrier_pct": fallback_profit_pct,
            "stop_barrier_pct": fallback_stop_pct,
            "barrier_basis": "fallback_fixed",
        }
    try:
        vol = float(daily_vol_pct)
    except (TypeError, ValueError):
        return {
            "profit_barrier_pct": fallback_profit_pct,
            "stop_barrier_pct": fallback_stop_pct,
            "barrier_basis": "fallback_fixed",
        }
    if vol <= 0:
        return {
            "profit_barrier_pct": fallback_profit_pct,
            "stop_barrier_pct": fallback_stop_pct,
            "barrier_basis": "fallback_fixed",
        }

    profit = min(max(vol * profit_mult, min_profit_pct), max_profit_pct)
    stop = min(max(vol * stop_mult, min_stop_pct), max_stop_pct)
    return {
        "profit_barrier_pct": round(profit, 2),
        "stop_barrier_pct": round(stop, 2),
        "barrier_basis": "volatility_scaled",
        "barrier_profit_mult": profit_mult,
        "barrier_stop_mult": stop_mult,
    }


def label_closed_trade(trade: Dict) -> Dict:
    """Assign a triple-barrier label to a closed position.

    Label semantics:
        +1  profit barrier touched first (TAKE_PROFIT, PROFIT_PROTECTION)
        -1  stop barrier touched first  (STOP_LOSS, TRAILING_STOP, CUT_LOSS)
         0  vertical barrier — time ran out with neither touched

    Early discretionary exits (STALE_POSITION, STRATEGY_RESET) are labelled 0 and
    flagged, because they aren't barrier events at all: the trade was terminated
    before the experiment finished, and treating them as genuine outcomes is what
    made the old history unusable for learning.
    """
    close_type = (trade.get("close_type") or "").upper()
    pnl_pct = trade.get("pnl_pct")

    profit_events = {"TAKE_PROFIT", "PROFIT_PROTECTION"}
    stop_events = {"STOP_LOSS", "TRAILING_STOP", "CUT_LOSS"}
    time_events = {"TIME_EXPIRED"}
    censored_events = {"STALE_POSITION", "STRATEGY_RESET"}

    if close_type in profit_events:
        label, basis, censored = 1, "profit_barrier", False
    elif close_type in stop_events:
        label, basis, censored = -1, "stop_barrier", False
    elif close_type in time_events:
        label, basis, censored = 0, "vertical_barrier", False
    elif close_type in censored_events:
        label, basis, censored = 0, "censored_early_exit", True
    else:
        label, basis, censored = 0, "unknown", True

    return {
        "tb_label": label,
        "tb_basis": basis,
        "tb_censored": censored,
        "tb_pnl_pct": pnl_pct,
    }


def is_trainable(labeled: Dict) -> bool:
    """Whether a labelled trade should feed the secondary model.

    Censored exits are excluded: they say more about the exit logic that was in
    force at the time than about whether the entry was any good.
    """
    return not labeled.get("tb_censored", True)
