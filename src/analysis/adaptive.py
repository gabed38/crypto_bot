"""
Adaptive segment learning — closing the loop on what actually works.

WHY THIS EXISTS, AND WHY IT IS NOT lessons.json
------------------------------------------------
This project already had a learning mechanism: after each batch of closes, an LLM
wrote prose lessons into `lessons.json`, and those lessons were injected into the
next run's prompt. It did not work. Reviewing months of history, the *same* lessons
recur near-verbatim — "do not cut early", "do not override hard rules with regime
statistics" — while the behaviour they warned about kept happening. Advisory text
in a prompt is not a control.

The difference here is mechanical enforcement. This module does not write advice
for a model to read and ignore. It buckets realised outcomes by segment, computes
a confidence interval on each segment's win rate, and *blocks* segments whose
performance is confidently below break-even. The gate is code. A model cannot talk
its way past it.

DESIGN NOTES
------------
Uses Wilson score intervals rather than raw win rates, because raw rates on small
samples are the exact failure mode that makes naive adaptive systems thrash: three
losses in a row is not evidence, and a system that reacts to it will chase noise.
A segment is only suppressed when the *upper* bound of its interval is below
break-even — that is, when even an optimistic reading of the evidence says it loses.

Observations decay exponentially (default 30-day half-life) so the system tracks a
changing market rather than being anchored to last quarter's regime. Decay makes
effective counts non-integer, which Wilson handles fine.

Break-even is a parameter, not 0.5, because it depends on payoff geometry: a trade
with an 8% target and a 16% stop needs a 2/3 win rate to break even, not half.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SEGMENTS_FILE = Path("data/performance/segments.json")

# Dimensions usable as an entry GATE. Every one of these must be knowable at
# decision time. Anything describing how a trade ended (close_type, pnl, exit
# reason) is leakage — it cannot inform a decision made before the trade exists.
GATING_DIMENSIONS = ("sector", "vol_signal", "time_horizon", "regime", "entry_type")

# Additional dimensions worth reporting on but NEVER gating on, because they are
# only known after the fact.
DIAGNOSTIC_DIMENSIONS = ("close_type",)

# z for the confidence interval. 1.28 ≈ 80% two-sided — deliberately not 95%.
# At 95% almost nothing reaches significance on these sample sizes and the system
# never acts; at 80% it acts on decent evidence while still ignoring noise.
DEFAULT_Z = 1.28


class SegmentTracker:
    """Tracks per-segment performance and suppresses confidently-losing segments."""

    def __init__(
        self,
        half_life_days: float = 30.0,
        min_samples: float = 8.0,
        z: float = DEFAULT_Z,
    ):
        self.half_life_days = half_life_days
        self.min_samples = min_samples
        self.z = z
        # key -> list of (iso_timestamp, won:int, pnl:float)
        self.observations: Dict[str, List[Tuple[str, int, float]]] = {}

    # ── recording ────────────────────────────────────────────────────────────

    @staticmethod
    def segment_keys(trade: Dict, dimensions: Tuple[str, ...]) -> List[str]:
        """Build segment keys from a trade, one per dimension.

        Each dimension is tracked independently ("regime=fear", "vol_signal=EXTREME")
        rather than as a combined key. Combined keys fragment the sample so finely
        that nothing ever reaches significance.
        """
        keys = []
        for dim in dimensions:
            val = trade.get(dim)
            if val is None or val == "":
                continue
            keys.append(f"{dim}={val}")
        return keys

    def record(
        self,
        trade: Dict,
        dimensions: Tuple[str, ...],
        won: bool,
        pnl: float = 0.0,
        when: Optional[datetime] = None,
    ) -> List[str]:
        """Record one closed trade against every segment it belongs to."""
        ts = (when or datetime.now()).isoformat()
        keys = self.segment_keys(trade, dimensions)
        for key in keys:
            self.observations.setdefault(key, []).append((ts, 1 if won else 0, float(pnl)))
        return keys

    # ── statistics ───────────────────────────────────────────────────────────

    def _decayed(self, key: str, now: Optional[datetime] = None) -> Tuple[float, float, float]:
        """Return (effective_n, effective_wins, total_pnl) with recency decay."""
        now = now or datetime.now()
        obs = self.observations.get(key, [])
        n_eff = 0.0
        w_eff = 0.0
        pnl_total = 0.0
        for ts, won, pnl in obs:
            try:
                age_days = (now - datetime.fromisoformat(ts)).total_seconds() / 86400
            except (TypeError, ValueError):
                age_days = 0.0
            weight = 0.5 ** (max(age_days, 0.0) / self.half_life_days)
            n_eff += weight
            w_eff += weight * won
            pnl_total += pnl
        return n_eff, w_eff, pnl_total

    def stats(self, key: str, now: Optional[datetime] = None) -> Optional[Dict]:
        """Wilson interval on the decayed win rate for one segment."""
        n_eff, w_eff, pnl = self._decayed(key, now)
        if n_eff <= 0:
            return None
        p = w_eff / n_eff
        lo, hi = _wilson_interval(p, n_eff, self.z)
        return {
            "segment": key,
            "n_raw": len(self.observations.get(key, [])),
            "n_effective": round(n_eff, 2),
            "win_rate": round(p, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "total_pnl": round(pnl, 2),
        }

    # ── the actual control ───────────────────────────────────────────────────

    def should_suppress(
        self,
        key: str,
        break_even: float = 0.5,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """Block a segment only when even an optimistic read says it loses.

        Returns (suppress, human-readable reason). Requires min_samples of
        *effective* (decayed) evidence, so a segment that performed badly a year ago
        stops being suppressed once that evidence ages out.
        """
        s = self.stats(key, now)
        if s is None:
            return False, "no data"
        if s["n_effective"] < self.min_samples:
            return False, f"insufficient evidence (n_eff={s['n_effective']:.1f} < {self.min_samples})"
        if s["ci_high"] < break_even:
            return True, (
                f"win rate {s['win_rate']:.0%} (n_eff={s['n_effective']:.1f}), "
                f"upper bound {s['ci_high']:.0%} < break-even {break_even:.0%}"
            )
        return False, f"within tolerance (upper bound {s['ci_high']:.0%} >= {break_even:.0%})"

    def evaluate_trade(
        self,
        trade: Dict,
        dimensions: Tuple[str, ...] = GATING_DIMENSIONS,
        break_even: float = 0.5,
        now: Optional[datetime] = None,
    ) -> Dict:
        """Check every segment a proposed trade belongs to.

        A trade is blocked if *any* of its segments is confidently losing — the
        blocking segment is named so the run log says why.

        Rejects post-hoc dimensions outright: gating on how past trades *ended*
        would be leakage, since that is unknowable for a trade not yet placed.
        """
        leaky = set(dimensions) & set(DIAGNOSTIC_DIMENSIONS)
        if leaky:
            raise ValueError(
                f"Cannot gate on post-hoc dimension(s) {sorted(leaky)} — "
                f"not knowable at entry. Use GATING_DIMENSIONS."
            )
        blocked_by = []
        for key in self.segment_keys(trade, dimensions):
            suppress, reason = self.should_suppress(key, break_even, now)
            if suppress:
                blocked_by.append({"segment": key, "reason": reason})
        return {
            "adaptive_blocked": bool(blocked_by),
            "adaptive_blocked_by": blocked_by,
        }

    def report(self, break_even: float = 0.5, now: Optional[datetime] = None) -> List[Dict]:
        """All segments with enough evidence, worst first."""
        rows = []
        for key in self.observations:
            s = self.stats(key, now)
            if s is None:
                continue
            suppress, reason = self.should_suppress(key, break_even, now)
            s["suppressed"] = suppress
            s["verdict"] = reason
            rows.append(s)
        rows.sort(key=lambda r: (not r["suppressed"], r["win_rate"]))
        return rows

    # ── persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "half_life_days": self.half_life_days,
            "min_samples": self.min_samples,
            "z": self.z,
            "observations": {k: [list(o) for o in v] for k, v in self.observations.items()},
        }

    def save(self, path: Path = SEGMENTS_FILE, variant: str = "default") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob: Dict = {}
        if path.exists():
            try:
                blob = json.loads(path.read_text())
            except Exception:
                blob = {}
        blob[variant] = self.to_dict()
        path.write_text(json.dumps(blob, indent=2))

    @classmethod
    def load(cls, path: Path = SEGMENTS_FILE, variant: str = "default") -> "SegmentTracker":
        if not path.exists():
            return cls()
        try:
            blob = json.loads(path.read_text())
        except Exception:
            return cls()
        data = blob.get(variant) or blob.get("default")
        if not data:
            return cls()
        t = cls(
            half_life_days=float(data.get("half_life_days", 30.0)),
            min_samples=float(data.get("min_samples", 8.0)),
            z=float(data.get("z", DEFAULT_Z)),
        )
        t.observations = {
            k: [(o[0], int(o[1]), float(o[2])) for o in v]
            for k, v in (data.get("observations") or {}).items()
        }
        return t

    @classmethod
    def rebuild_from_history(
        cls,
        trades: List[Dict],
        dimensions: Tuple[str, ...],
        outcome_fn,
        timestamp_field: str = "resolved_at",
        **kwargs,
    ) -> "SegmentTracker":
        """Build a tracker from scratch out of resolved-trade history.

        Used by refit_models.py so the tracker reflects all history rather than only
        what accumulated since it was switched on.
        """
        t = cls(**kwargs)
        for trade in trades:
            outcome = outcome_fn(trade)
            if outcome is None:
                continue
            when = None
            ts = trade.get(timestamp_field)
            if ts:
                try:
                    when = datetime.fromisoformat(str(ts))
                except (TypeError, ValueError):
                    when = None
            t.record(trade, dimensions, bool(outcome), trade.get("pnl_usd") or 0.0, when)
        return t


def _wilson_interval(p: float, n: float, z: float) -> Tuple[float, float]:
    """Wilson score interval — well-behaved at small n and at p near 0 or 1,
    where the normal approximation produces nonsense like negative bounds."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(max(p * (1 - p) / n + (z * z) / (4 * n * n), 0.0))
    return max(0.0, centre - margin), min(1.0, centre + margin)
