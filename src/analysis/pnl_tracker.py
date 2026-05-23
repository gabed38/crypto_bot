"""PnL tracking and financial analysis across resolved crypto trades."""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict


RESOLVED_FILE = Path("data/positions/resolved_trades.jsonl")


def _amount_invested(trade: Dict) -> float:
    if trade.get("amount_invested") is not None:
        return float(trade["amount_invested"])
    return 5.0  # default flat $5


def _pnl_usd(trade: Dict) -> Optional[float]:
    pnl_pct = trade.get("pnl_pct")
    if pnl_pct is None:
        return None
    return round((pnl_pct / 100.0) * _amount_invested(trade), 2)


class PnLTracker:
    """Aggregate trade P&L by day, week, and month."""

    def __init__(self, resolved_file: Path = RESOLVED_FILE):
        self.resolved_file = resolved_file

    def load_resolved(self, days: Optional[int] = None) -> List[Dict]:
        if not self.resolved_file.exists():
            return []
        cutoff = datetime.now() - timedelta(days=days) if days else None
        trades = []
        try:
            with open(self.resolved_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        if cutoff:
                            ts = t.get("resolved_at") or t.get("executed_at", "")
                            if ts:
                                try:
                                    if datetime.fromisoformat(ts) < cutoff:
                                        continue
                                except Exception:
                                    pass
                        t["amount_invested"] = _amount_invested(t)
                        if t.get("pnl_usd") is None:
                            t["pnl_usd"] = _pnl_usd(t)
                        trades.append(t)
                    except Exception:
                        continue
        except Exception:
            pass
        return trades

    def _period_key(self, dt: datetime, period: str) -> str:
        if period == "day":
            return dt.strftime("%Y-%m-%d")
        if period == "week":
            iso = dt.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        if period == "month":
            return dt.strftime("%Y-%m")
        return dt.strftime("%Y-%m-%d")

    def _group(self, trades: List[Dict], period: str) -> Dict[str, Dict]:
        groups: Dict[str, List] = defaultdict(list)
        for t in trades:
            ts = t.get("resolved_at") or t.get("executed_at") or t.get("execution_timestamp", "")
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            groups[self._period_key(dt, period)].append(t)

        result = {}
        for key, bucket in sorted(groups.items()):
            wins = [t for t in bucket if t.get("trade_result") == "WIN"]
            losses = [t for t in bucket if t.get("trade_result") == "LOSS"]
            resolved = wins + losses
            pnl = sum(t["pnl_usd"] for t in bucket if t.get("pnl_usd") is not None)
            invested = sum(t["amount_invested"] for t in bucket)
            result[key] = {
                "trades": len(bucket),
                "wins": len(wins),
                "losses": len(losses),
                "closed_early": len([t for t in bucket if t.get("trade_result") == "CLOSED_EARLY"]),
                "win_rate_pct": round(len(wins) / len(resolved) * 100, 1) if resolved else None,
                "pnl_usd": round(pnl, 2),
                "amount_invested": round(invested, 2),
                "roi_pct": round(pnl / invested * 100, 1) if invested > 0 else None,
            }
        return result

    def daily_summary(self, days: int = 30) -> Dict[str, Dict]:
        return self._group(self.load_resolved(days=days), "day")

    def weekly_summary(self, weeks: int = 12) -> Dict[str, Dict]:
        return self._group(self.load_resolved(days=weeks * 7), "week")

    def monthly_summary(self, months: int = 12) -> Dict[str, Dict]:
        return self._group(self.load_resolved(days=months * 30), "month")

    def all_time_summary(self) -> Dict[str, Any]:
        trades = self.load_resolved()
        wins = [t for t in trades if t.get("trade_result") == "WIN"]
        losses = [t for t in trades if t.get("trade_result") == "LOSS"]
        resolved = wins + losses
        pnl = sum(t["pnl_usd"] for t in trades if t.get("pnl_usd") is not None)
        invested = sum(t["amount_invested"] for t in trades)
        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(resolved) * 100, 1) if resolved else None,
            "total_pnl_usd": round(pnl, 2),
            "total_invested": round(invested, 2),
            "roi_pct": round(pnl / invested * 100, 1) if invested > 0 else None,
        }

    def per_trade_report(self, days: Optional[int] = None) -> List[Dict]:
        trades = self.load_resolved(days=days)
        return [
            {
                "date": (t.get("resolved_at") or t.get("executed_at") or "")[:10],
                "symbol": t.get("symbol", "?"),
                "coin_name": t.get("coin_name", "?"),
                "direction": t.get("direction", "LONG"),
                "result": t.get("trade_result", "?"),
                "entry_price": t.get("entry_price"),
                "exit_price": t.get("close_price"),
                "amount_invested": t["amount_invested"],
                "pnl_usd": t.get("pnl_usd"),
                "pnl_pct": t.get("pnl_pct"),
            }
            for t in trades
        ]

    def format_report(self) -> str:
        lines = []
        all_time = self.all_time_summary()
        wr = f"{all_time['win_rate_pct']}%" if all_time["win_rate_pct"] is not None else "N/A"
        roi = f"{all_time['roi_pct']}%" if all_time["roi_pct"] is not None else "N/A"
        lines += [
            "ALL-TIME FINANCIAL SUMMARY",
            f"  Total trades    : {all_time['total_trades']}  ({all_time['wins']}W / {all_time['losses']}L)",
            f"  Win rate        : {wr}",
            f"  Total P&L       : ${all_time['total_pnl_usd']:+.2f}",
            f"  Total invested  : ${all_time['total_invested']:.2f}",
            f"  ROI             : {roi}",
        ]

        monthly = self.monthly_summary()
        if monthly:
            lines.append("\nMONTHLY P&L")
            for month, s in sorted(monthly.items(), reverse=True):
                wr = f"{s['win_rate_pct']}%" if s["win_rate_pct"] is not None else "N/A"
                roi = f"{s['roi_pct']}%" if s["roi_pct"] is not None else "N/A"
                lines.append(
                    f"  {month}: ${s['pnl_usd']:+.2f}  "
                    f"({s['wins']}W/{s['losses']}L  win rate: {wr}  ROI: {roi}  "
                    f"invested: ${s['amount_invested']:.0f})"
                )

        daily = self.daily_summary(days=14)
        if daily:
            lines.append("\nDAILY P&L (last 14 days)")
            for day, s in sorted(daily.items(), reverse=True):
                wr = f"{s['win_rate_pct']}%" if s["win_rate_pct"] is not None else "N/A"
                lines.append(f"  {day}: ${s['pnl_usd']:+.2f}  ({s['wins']}W/{s['losses']}L  win rate: {wr})")

        return "\n".join(lines)
