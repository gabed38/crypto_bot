"""
Shared helper functions for crypto trading bot.

Position I/O, evaluation, lesson storage, and prompt building.
"""

import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.utils.config import Config

POSITIONS_FILE = Path("data/positions/open_positions.json")
RESOLVED_FILE = Path("data/positions/resolved_trades.jsonl")
LESSONS_FILE = Path("data/performance/lessons.json")


# ── position file helpers ─────────────────────────────────────────────────────

def load_open_positions() -> List[Dict]:
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_open_positions(positions: List[Dict]) -> None:
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def add_to_open_positions(trades: List[Dict]) -> None:
    """Add newly executed trades to the open positions file, skipping duplicates."""
    existing = load_open_positions()
    held_ids = {p.get("coin_id", "") + p.get("direction", "LONG") for p in existing}
    fields = (
        # Core identity
        "coin_id", "symbol", "coin_name", "direction",
        # Financials
        "entry_price", "amount_invested", "conviction",
        "target_pct", "stop_loss_pct",
        # Context
        "reasoning", "risks", "time_horizon",
        "execution_date", "execution_timestamp",
        # Quant screen enrichments
        "screen_score", "daily_vol_pct", "stop_multiple", "vol_signal",
        "adaptive_stop_pct", "sector",
        # Market snapshot at entry
        "market_cap", "volume_24h", "price_change_24h",
    )
    for trade in trades:
        key = trade.get("coin_id", "") + trade.get("direction", "LONG")
        if key and key not in held_ids:
            existing.append({k: trade.get(k) for k in fields})
            held_ids.add(key)
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def append_resolved_trade(result: Dict) -> None:
    RESOLVED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESOLVED_FILE, "a") as f:
        f.write(json.dumps({**result, "resolved_at": datetime.now().isoformat()}) + "\n")


def load_recent_resolved(days: int = 14) -> List[Dict]:
    if not RESOLVED_FILE.exists():
        return []
    cutoff = datetime.now() - timedelta(days=days)
    trades = []
    try:
        with open(RESOLVED_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    resolved_at = t.get("resolved_at", "")
                    if resolved_at:
                        dt = datetime.fromisoformat(resolved_at)
                        if dt >= cutoff:
                            trades.append(t)
                except Exception:
                    continue
    except Exception:
        pass
    return trades


# ── time-horizon helpers ─────────────────────────────────────────────────────

def _parse_horizon_days(horizon_str: str) -> Optional[int]:
    """Parse a time-horizon string to integer days.

    Supported formats: '1d', '7d', '2w', '1m' (d=days, w=weeks, m=months≈30d).
    Returns None if the string cannot be parsed.
    """
    m = re.match(r"^(\d+)(d|w|m)$", str(horizon_str).lower().strip())
    if not m:
        return None
    value, unit = int(m.group(1)), m.group(2)
    return {"d": value, "w": value * 7, "m": value * 30}[unit]


def check_time_horizon_expired(position: Dict) -> bool:
    """Return True if the position has been held at least as long as its time_horizon.

    Uses execution_date (YYYY-MM-DD) if present, falls back to execution_timestamp.
    Returns False when either field is missing or unparseable.
    """
    exec_str = (
        position.get("execution_date")
        or position.get("execution_timestamp", "")
    )
    if not exec_str:
        return False

    horizon_days = _parse_horizon_days(position.get("time_horizon", ""))
    if horizon_days is None:
        return False

    try:
        opened = datetime.fromisoformat(str(exec_str)[:10])
        days_held = (datetime.now() - opened).days
        return days_held >= horizon_days
    except Exception:
        return False


# ── position evaluation ──────────────────────────────────────────────────────

def evaluate_position(position: Dict, current_price: Optional[float]) -> Dict:
    """
    Evaluate an open crypto position against its current price.

    Returns a result dict with pnl_pct, pnl_usd, and latest_price.
    Unlike Polymarket, crypto positions don't "resolve" — they are only
    closed by the bot (stop-loss, take-profit, or cut-loss).
    """
    entry_price = position.get("entry_price", 0)
    direction = position.get("direction", "LONG")
    amount_invested = position.get("amount_invested", 5.0)

    result = {
        **position,
        "status": "open",
        "latest_price": current_price,
        "pnl_pct": None,
        "pnl_usd": None,
        "trade_result": "UNREALIZED",
    }

    if current_price is None or not entry_price:
        result["status"] = "price_unavailable"
        return result

    if direction == "LONG":
        pnl_pct = round((current_price - entry_price) / entry_price * 100, 2)
    else:
        pnl_pct = round((entry_price - current_price) / entry_price * 100, 2)

    result["pnl_pct"] = pnl_pct
    result["pnl_usd"] = round((pnl_pct / 100.0) * amount_invested, 2)

    return result


# ── lessons store ────────────────────────────────────────────────────────────

def save_lessons(new_entry: Dict) -> None:
    """Save a lesson entry to the lessons file (single list, not keyed by niche)."""
    LESSONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    lessons: List = []
    if LESSONS_FILE.exists():
        try:
            with open(LESSONS_FILE) as f:
                lessons = json.load(f)
        except Exception:
            pass
    lessons.append(new_entry)
    lessons = lessons[-60:]  # keep last 60 entries
    with open(LESSONS_FILE, "w") as f:
        json.dump(lessons, f, indent=2)


def load_recent_lessons(max_entries: int = 5) -> str:
    """Load recent lessons and format for LLM injection."""
    if not LESSONS_FILE.exists():
        return ""
    try:
        with open(LESSONS_FILE) as f:
            lessons = json.load(f)
    except Exception:
        return ""

    if not lessons:
        return ""

    recent = lessons[-max_entries:]
    lines = ["\n--- LESSONS FROM RECENT SESSIONS (use these to improve your analysis) ---"]
    for entry in recent:
        date = entry.get("date", "?")
        wr = entry.get("win_rate_pct")
        pnl = entry.get("pnl_usd")
        lines.append(f"\nSession {date} (Win rate: {wr}%, P&L: ${pnl:+.2f}):")

        for lesson in entry.get("lessons", []):
            lines.append(f"  - {lesson}")
        for item in entry.get("what_worked", []):
            lines.append(f"  [WORKED] {item}")
        for item in entry.get("what_didnt_work", []):
            lines.append(f"  [FAILED] {item}")

    return "\n".join(lines)


# ── hold/close prompt builder ────────────────────────────────────────────────

def build_hold_close_prompt(positions_with_prices: List[Dict], recent_resolved: List[Dict]) -> str:
    """Build the LLM prompt for hold/close analysis on open positions."""
    lines = []

    if positions_with_prices:
        lines.append("CURRENT OPEN POSITIONS (assess each for HOLD or CLOSE):")
        for r in positions_with_prices:
            entry = r.get("entry_price", 0)
            latest = r.get("latest_price", 0)
            pnl_str = f"{r['pnl_pct']:+.1f}%" if r.get("pnl_pct") is not None else "N/A"
            pnl_usd = f"${r['pnl_usd']:+.2f}" if r.get("pnl_usd") is not None else "N/A"
            direction = r.get("direction", "LONG")

            adaptive_stop = r.get("adaptive_stop_pct", "?")
            target_pct = r.get("target_pct", "?")
            stop_str = f"{adaptive_stop:.1f}%" if isinstance(adaptive_stop, float) else adaptive_stop
            target_str = f"{target_pct:.1f}%" if isinstance(target_pct, float) else target_pct
            lines.append(
                f"\n  Coin: {r.get('coin_name', '?')} ({r.get('symbol', '?').upper()})"
                f"\n    Coin ID: {r.get('coin_id', '?')}"
                f"\n    Direction: {direction} | Entry: ${entry:,.4f} | Now: ${latest:,.4f}"
                f"\n    P&L: {pnl_str} ({pnl_usd})"
                f"\n    Target: {target_str} | Vol-stop: {stop_str}"
                f"\n    Invested: ${r.get('amount_invested', 5):.2f} | Conviction: {r.get('conviction', '?')}"
                f"\n    Opened: {r.get('execution_date', '?')} | Time horizon: {r.get('time_horizon', '?')}"
                f"\n    Reasoning at entry: {r.get('reasoning', '')[:200]}"
            )

    if recent_resolved:
        lines.append(f"\n\nRECENT CLOSED TRADES (last 14 days):")
        for r in recent_resolved[-10:]:
            pnl_str = f"{r.get('pnl_pct', 0):+.1f}%" if r.get("pnl_pct") is not None else "N/A"
            lines.append(
                f"  [{r.get('trade_result', '?')}] {r.get('symbol', '?').upper()} "
                f"({r.get('direction', 'LONG')}) | P&L: {pnl_str}"
            )

    trades_text = "\n".join(lines) if lines else "No positions to evaluate."

    return f"""You are reviewing the open positions of a crypto trading bot.

{trades_text}

For each open position, decide: HOLD or CLOSE.

PROFIT-TAKING PHILOSOPHY (volatile markets):
In highly volatile crypto markets, locking in gains quickly beats holding for maximum upside.
If a position is near or past its target_pct, take the profit — don't wait for more.
A 5% gain taken now is worth more than a 10% gain that reverts to 0%.

Reasons to CLOSE (take profit):
- P&L is at or above the original target_pct — take the win, don't get greedy
- The coin has made a sharp move and momentum is slowing (reversal likely)
- Time horizon has elapsed or is about to — exit cleanly
- Broader market turning bearish (fear increasing, BTC dropping)

Reasons to CLOSE (cut loss):
- The original thesis has been invalidated (bad news, regulatory action, hack, etc.)
- The coin has broken below key support and is trending down
- Loss exceeds 20% and no specific catalyst for recovery
- Time horizon exceeded with no progress toward target

Reasons to HOLD:
- P&L is below target and momentum is still intact
- Time horizon still has room and thesis hasn't changed
- Strong tailwinds (BTC pumping, positive macro sentiment)

Format your response as JSON:
{{
  "positions_to_close": [
    {{
      "coin_id": "...",
      "action": "CLOSE",
      "close_type": "TAKE_PROFIT or CUT_LOSS",
      "reason": "One sentence explaining the close decision"
    }}
  ],
  "positions_to_hold": [
    {{
      "coin_id": "...",
      "action": "HOLD",
      "reason": "One sentence explaining why to hold"
    }}
  ],
  "market_assessment": "Brief overall crypto market assessment"
}}"""
