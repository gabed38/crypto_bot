"""
Probability calibration and proper scoring.

Motivation: both bots emit a "conviction" score that was being used as if it were
a probability. Measured against resolved trades it wasn't one — stated 0.75
conviction delivered a 25% win rate, stated 0.70 delivered 52%. This module turns
a raw score into a calibrated probability and provides the proper scoring rules
needed to tell whether that mapping is actually working.

Platt scaling (a one-feature logistic fit) is used rather than isotonic regression
because it has two parameters and stays stable on the few-hundred-sample histories
these bots have. Switch to isotonic once there are several hundred labels per
variant — see `PlattCalibrator.recommend_method`.

Pure stdlib: no numpy/sklearn dependency, so this runs anywhere the bot runs.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

CALIBRATION_FILE = Path("data/performance/calibration.json")


# ── proper scoring rules ─────────────────────────────────────────────────────

def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> Optional[float]:
    """Mean squared error between forecast probability and binary outcome.

    Lower is better. 0.25 is what you get by always saying 50%; a score above
    that means the forecasts are worse than useless. Unlike win rate, this
    rewards being confident only when confidence is warranted.
    """
    pairs = _clean_pairs(probabilities, outcomes)
    if not pairs:
        return None
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def log_loss(probabilities: Sequence[float], outcomes: Sequence[int], eps: float = 1e-9) -> Optional[float]:
    """Logarithmic scoring rule. Punishes confident wrong answers far harder
    than Brier does — useful as a secondary check when a model is directionally
    fine but badly overconfident."""
    pairs = _clean_pairs(probabilities, outcomes)
    if not pairs:
        return None
    total = 0.0
    for p, o in pairs:
        p = min(max(p, eps), 1 - eps)
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(pairs)


def reference_brier(outcomes: Sequence[int]) -> Optional[float]:
    """Brier score of always forecasting the base rate.

    This is the number to beat. A model that can't beat its own base rate is
    contributing nothing beyond knowing how often things happen in general.
    """
    outs = [int(o) for o in outcomes if o is not None]
    if not outs:
        return None
    base = sum(outs) / len(outs)
    return sum((base - o) ** 2 for o in outs) / len(outs)


def skill_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> Optional[float]:
    """Brier skill score: 1 - (model Brier / base-rate Brier).

    Positive means the model beats always-guess-the-base-rate. Negative means it
    is actively worse than that, which is the situation both bots were in.
    """
    bs = brier_score(probabilities, outcomes)
    ref = reference_brier(outcomes)
    if bs is None or ref is None or ref == 0:
        return None
    return 1 - (bs / ref)


def reliability_curve(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Bucket forecasts and compare stated probability to observed frequency.

    Returns one row per non-empty bin with the mean forecast, the realised
    frequency, and the gap. A well-calibrated model has gap ≈ 0 everywhere.
    """
    pairs = _clean_pairs(probabilities, outcomes)
    if not pairs:
        return []

    bins: Dict[int, List[Tuple[float, int]]] = {}
    for p, o in pairs:
        idx = min(int(p * n_bins), n_bins - 1)
        bins.setdefault(idx, []).append((p, o))

    rows = []
    for idx in sorted(bins):
        members = bins[idx]
        mean_p = sum(p for p, _ in members) / len(members)
        freq = sum(o for _, o in members) / len(members)
        rows.append({
            "bin_low": idx / n_bins,
            "bin_high": (idx + 1) / n_bins,
            "n": len(members),
            "mean_forecast": round(mean_p, 4),
            "observed_freq": round(freq, 4),
            "gap": round(freq - mean_p, 4),
        })
    return rows


def expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> Optional[float]:
    """Sample-weighted mean absolute gap across reliability bins."""
    curve = reliability_curve(probabilities, outcomes, n_bins)
    if not curve:
        return None
    total = sum(row["n"] for row in curve)
    if total == 0:
        return None
    return sum(row["n"] * abs(row["gap"]) for row in curve) / total


# ── Platt scaling ────────────────────────────────────────────────────────────

class PlattCalibrator:
    """Maps a raw score (e.g. LLM conviction) to a calibrated probability.

    Fits p = sigmoid(a * score + b) by gradient descent on log loss. Two
    parameters keeps it stable on small samples — the whole point, given these
    bots have tens-to-low-hundreds of resolved trades.

    Until fitted, `transform` is the identity: callers get the raw score back,
    so wiring this in never silently changes behaviour before there's data to
    justify a change.
    """

    MIN_SAMPLES = 30

    def __init__(self, a: float = 1.0, b: float = 0.0, fitted: bool = False):
        self.a = a
        self.b = b
        self.fitted = fitted
        self.n_samples = 0
        self.fit_metrics: Dict[str, Optional[float]] = {}

    # -- fitting --

    def fit(
        self,
        scores: Sequence[float],
        outcomes: Sequence[int],
        lr: float = 0.1,
        epochs: int = 4000,
    ) -> "PlattCalibrator":
        """Fit on historical (score, outcome) pairs.

        Refuses to fit below MIN_SAMPLES — an undersized fit is worse than no
        fit, because it produces confident-looking garbage.
        """
        pairs = _clean_pairs(scores, outcomes, clamp=False)
        if len(pairs) < self.MIN_SAMPLES:
            self.fitted = False
            self.n_samples = len(pairs)
            return self

        a, b = 1.0, 0.0
        n = len(pairs)
        for _ in range(epochs):
            grad_a = 0.0
            grad_b = 0.0
            for s, o in pairs:
                pred = _sigmoid(a * s + b)
                err = pred - o
                grad_a += err * s
                grad_b += err
            a -= lr * grad_a / n
            b -= lr * grad_b / n

        self.a, self.b = a, b
        self.fitted = True
        self.n_samples = n

        calibrated = [self.transform(s) for s, _ in pairs]
        outs = [o for _, o in pairs]
        raw = [min(max(s, 0.0), 1.0) for s, _ in pairs]
        self.fit_metrics = {
            "brier_raw": brier_score(raw, outs),
            "brier_calibrated": brier_score(calibrated, outs),
            "ece_raw": expected_calibration_error(raw, outs),
            "ece_calibrated": expected_calibration_error(calibrated, outs),
            "skill_calibrated": skill_score(calibrated, outs),
            "base_rate": sum(outs) / len(outs),
        }
        return self

    def transform(self, score: float) -> float:
        """Map a raw score to a calibrated probability (identity until fitted)."""
        if score is None:
            return 0.5
        if not self.fitted:
            return min(max(float(score), 0.0), 1.0)
        return _sigmoid(self.a * float(score) + self.b)

    def improves_on_raw(self) -> bool:
        """True when the fitted mapping actually scores better than the raw score.

        Guards against adopting a calibration that makes things worse — which is
        possible when the underlying score carries no signal at all.
        """
        if not self.fitted:
            return False
        raw = self.fit_metrics.get("brier_raw")
        cal = self.fit_metrics.get("brier_calibrated")
        if raw is None or cal is None:
            return False
        return cal < raw

    @staticmethod
    def recommend_method(n_samples: int) -> str:
        """Which calibration family suits this sample size."""
        if n_samples < PlattCalibrator.MIN_SAMPLES:
            return "none — too few samples, use raw score"
        if n_samples < 300:
            return "platt — two parameters, stable at this size"
        return "isotonic — enough labels to fit a free-form monotone mapping"

    # -- persistence --

    def to_dict(self) -> Dict:
        return {
            "a": self.a,
            "b": self.b,
            "fitted": self.fitted,
            "n_samples": self.n_samples,
            "fit_metrics": self.fit_metrics,
            "recommended_method": self.recommend_method(self.n_samples),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlattCalibrator":
        c = cls(
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 0.0)),
            fitted=bool(data.get("fitted", False)),
        )
        c.n_samples = int(data.get("n_samples", 0))
        c.fit_metrics = data.get("fit_metrics", {}) or {}
        return c

    def save(self, path: Path = CALIBRATION_FILE, variant: str = "default") -> None:
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
    def load(cls, path: Path = CALIBRATION_FILE, variant: str = "default") -> "PlattCalibrator":
        """Load a fitted calibrator, falling back to an unfitted identity mapping."""
        if not path.exists():
            return cls()
        try:
            blob = json.loads(path.read_text())
        except Exception:
            return cls()
        if variant in blob:
            return cls.from_dict(blob[variant])
        if "default" in blob:
            return cls.from_dict(blob["default"])
        return cls()


# ── internals ────────────────────────────────────────────────────────────────

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _clean_pairs(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    clamp: bool = True,
) -> List[Tuple[float, int]]:
    """Drop rows where either side is missing/unparseable; optionally clamp to [0,1]."""
    pairs = []
    for p, o in zip(probabilities, outcomes):
        if p is None or o is None:
            continue
        try:
            pv = float(p)
            ov = int(bool(o)) if not isinstance(o, (int, float)) else int(o)
        except (TypeError, ValueError):
            continue
        if math.isnan(pv):
            continue
        if ov not in (0, 1):
            continue
        pairs.append((min(max(pv, 0.0), 1.0) if clamp else pv, ov))
    return pairs
