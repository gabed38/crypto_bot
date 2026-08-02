"""
Meta-labeling: the secondary model.

López de Prado's split separates two questions that the bot was answering with a
single LLM call:

    primary model   — which way, and on what? (the quant screen already does this)
    secondary model — is this particular signal worth acting on, and how big?

Collapsing both into one call is why `screen_score` currently has no relationship
to outcomes: nothing in the system was ever trained or measured on the narrower
question of "given this candidate, does taking it work out?"

This module is the secondary model. It has two backends:

    llm         — the LLM judges each screened candidate (bootstrap mode)
    classifier  — a logistic model fitted on triple-barrier labels

It starts on `llm` and switches to `classifier` automatically once enough
uncensored labels exist, because a classifier fitted on 30 rows is worse than no
classifier at all. Censored exits (positions killed early by discretionary rules)
are excluded from training — they describe the exit logic of the day, not whether
the entry had merit.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MODEL_FILE = Path("data/performance/meta_model.json")

# Screen outputs available at decision time. Anything computed after entry would
# leak the outcome into the features, so the list is deliberately narrow.
FEATURES = (
    "screen_score",
    "daily_vol_pct",
    "stop_multiple",
    "price_change_24h",
    "rsi",
    "vol_mcap_ratio",
)


class MetaLabeler:
    """Decides take/skip on candidates the primary model has already selected."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        bias: float = 0.0,
        fitted: bool = False,
        min_training_labels: int = 100,
    ):
        self.weights = weights or {}
        self.bias = bias
        self.fitted = fitted
        self.min_training_labels = min_training_labels
        self.n_samples = 0
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
        self.fit_metrics: Dict[str, Optional[float]] = {}

    # ── backend selection ────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return "classifier" if self.fitted else "llm"

    def status(self, available_labels: int) -> str:
        """Human-readable readiness line for the run log."""
        if self.fitted:
            return (
                f"classifier active (fitted on {self.n_samples} labels, "
                f"AUC {self.fit_metrics.get('auc', float('nan')):.3f})"
            )
        return (
            f"LLM secondary (need {self.min_training_labels} uncensored labels "
            f"to fit a classifier, have {available_labels})"
        )

    # ── feature extraction ───────────────────────────────────────────────────

    @staticmethod
    def extract_features(candidate: Dict) -> Dict[str, float]:
        """Pull the decision-time feature vector from a screened candidate."""
        mcap = _num(candidate.get("market_cap"))
        vol = _num(candidate.get("volume_24h"))
        vol_mcap = (vol / mcap) if (mcap and vol and mcap > 0) else 0.0
        return {
            "screen_score": _num(candidate.get("screen_score")) or 0.0,
            "daily_vol_pct": _num(candidate.get("daily_vol_pct")) or 0.0,
            "stop_multiple": _num(candidate.get("stop_multiple")) or 0.0,
            "price_change_24h": _num(candidate.get("price_change_24h")) or 0.0,
            "rsi": _num(candidate.get("rsi")) or 50.0,
            "vol_mcap_ratio": vol_mcap,
        }

    # ── training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        labeled_trades: Sequence[Dict],
        lr: float = 0.1,
        epochs: int = 3000,
    ) -> "MetaLabeler":
        """Fit on triple-barrier-labelled history.

        Expects records carrying `tb_label` and `tb_censored` from
        `src.analysis.labeling`. Only uncensored rows are used, and the target is
        "did the profit barrier get hit first" (label +1), not raw P&L sign.
        """
        rows: List[Tuple[Dict[str, float], int]] = []
        for t in labeled_trades:
            if t.get("tb_censored", True):
                continue
            label = t.get("tb_label")
            if label is None:
                continue
            rows.append((self.extract_features(t), 1 if label == 1 else 0))

        self.n_samples = len(rows)
        if len(rows) < self.min_training_labels:
            self.fitted = False
            return self

        # Standardise so one large-magnitude feature can't dominate the fit
        self.feature_means, self.feature_stds = _standardisation_params(rows)
        X = [_standardise(f, self.feature_means, self.feature_stds) for f, _ in rows]
        y = [label for _, label in rows]

        w = {k: 0.0 for k in FEATURES}
        b = 0.0
        n = len(X)
        for _ in range(epochs):
            gw = {k: 0.0 for k in FEATURES}
            gb = 0.0
            for xi, yi in zip(X, y):
                z = b + sum(w[k] * xi.get(k, 0.0) for k in FEATURES)
                pred = _sigmoid(z)
                err = pred - yi
                for k in FEATURES:
                    gw[k] += err * xi.get(k, 0.0)
                gb += err
            for k in FEATURES:
                w[k] -= lr * gw[k] / n
            b -= lr * gb / n

        self.weights, self.bias, self.fitted = w, b, True
        preds = [self.predict_proba_features(f) for f, _ in rows]
        self.fit_metrics = {
            "auc": _auc(preds, y),
            "base_rate": sum(y) / len(y),
            "mean_pred": sum(preds) / len(preds),
        }
        return self

    # ── prediction ───────────────────────────────────────────────────────────

    def predict_proba_features(self, features: Dict[str, float]) -> float:
        if not self.fitted:
            return 0.5
        x = _standardise(features, self.feature_means, self.feature_stds)
        z = self.bias + sum(self.weights.get(k, 0.0) * x.get(k, 0.0) for k in FEATURES)
        return _sigmoid(z)

    def predict_proba(self, candidate: Dict) -> float:
        return self.predict_proba_features(self.extract_features(candidate))

    def decide(self, candidate: Dict, threshold: float = 0.5) -> Dict:
        """Take/skip decision for one candidate.

        In LLM mode this defers — it returns the LLM's own conviction as the
        confidence and marks the source, so the pipeline behaves as before while
        still recording the decision in the shape the classifier will later fill.
        """
        if not self.fitted:
            conviction = _num(candidate.get("conviction")) or 0.0
            return {
                "meta_decision": "take" if conviction >= threshold else "skip",
                "meta_confidence": round(conviction, 4),
                "meta_source": "llm",
            }
        p = self.predict_proba(candidate)
        return {
            "meta_decision": "take" if p >= threshold else "skip",
            "meta_confidence": round(p, 4),
            "meta_source": "classifier",
        }

    # ── persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "fitted": self.fitted,
            "n_samples": self.n_samples,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "fit_metrics": self.fit_metrics,
            "min_training_labels": self.min_training_labels,
        }

    def save(self, path: Path = MODEL_FILE, variant: str = "default") -> None:
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
    def load(
        cls,
        path: Path = MODEL_FILE,
        variant: str = "default",
        min_training_labels: int = 100,
    ) -> "MetaLabeler":
        if not path.exists():
            return cls(min_training_labels=min_training_labels)
        try:
            blob = json.loads(path.read_text())
        except Exception:
            return cls(min_training_labels=min_training_labels)
        data = blob.get(variant) or blob.get("default")
        if not data:
            return cls(min_training_labels=min_training_labels)
        m = cls(
            weights=data.get("weights", {}),
            bias=float(data.get("bias", 0.0)),
            fitted=bool(data.get("fitted", False)),
            min_training_labels=int(data.get("min_training_labels", min_training_labels)),
        )
        m.n_samples = int(data.get("n_samples", 0))
        m.feature_means = data.get("feature_means", {}) or {}
        m.feature_stds = data.get("feature_stds", {}) or {}
        m.fit_metrics = data.get("fit_metrics", {}) or {}
        return m


# ── internals ────────────────────────────────────────────────────────────────

def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _standardisation_params(
    rows: Sequence[Tuple[Dict[str, float], int]]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    means, stds = {}, {}
    for k in FEATURES:
        vals = [f.get(k, 0.0) for f, _ in rows]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        means[k] = m
        stds[k] = math.sqrt(var) or 1.0
    return means, stds


def _standardise(
    features: Dict[str, float],
    means: Dict[str, float],
    stds: Dict[str, float],
) -> Dict[str, float]:
    return {
        k: (features.get(k, 0.0) - means.get(k, 0.0)) / (stds.get(k) or 1.0)
        for k in FEATURES
    }


def _auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Area under ROC via pairwise ranking. 0.5 = no discrimination."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))
