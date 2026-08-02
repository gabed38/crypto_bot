#!/usr/bin/env python3
"""
Refit the calibrator and secondary model from resolved-trade history, and report
calibration quality per strategy variant.

Run this periodically (weekly is plenty) and after any batch of closes:

    venv/bin/python scripts/refit_models.py            # report + refit
    venv/bin/python scripts/refit_models.py --dry-run  # report only

Why this exists: win rate is the metric that let a bot with a 51%-accurate signal
look healthy. Brier score and skill score don't allow that — skill score is
negative whenever the model is doing worse than simply knowing the base rate.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analysis.calibration import (
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    log_loss,
    reference_brier,
    reliability_curve,
    skill_score,
)
from src.analysis.labeling import label_closed_trade
from src.analysis.meta_labeler import MetaLabeler
from src.analysis.adaptive import SegmentTracker, GATING_DIMENSIONS, DIAGNOSTIC_DIMENSIONS

RESOLVED_FILE = Path("data/positions/resolved_trades.jsonl")


def _apply_config_paths(config_path: str) -> None:
    """Point RESOLVED_FILE at whichever variant's history this config owns."""
    global RESOLVED_FILE
    try:
        import os
        import yaml
        from string import Template
        raw = Path(config_path).read_text()
        cfg = yaml.safe_load(Template(raw).safe_substitute(os.environ)) or {}
    except Exception:
        return
    pos_dir = (cfg.get("data") or {}).get("positions_dir")
    if pos_dir:
        RESOLVED_FILE = Path(pos_dir) / "resolved_trades.jsonl"
    print(f"Reading resolved trades from: {RESOLVED_FILE}")


def load_resolved() -> List[Dict]:
    if not RESOLVED_FILE.exists():
        return []
    out = []
    with open(RESOLVED_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            t.update(label_closed_trade(t))
            out.append(t)
    return out


def outcome_of(trade: Dict) -> int:
    """Binary outcome for calibration: did the profit barrier get hit first?"""
    return 1 if trade.get("tb_label") == 1 else 0


def report_variant(name: str, trades: List[Dict]) -> None:
    scores = [t.get("conviction") for t in trades]
    outcomes = [outcome_of(t) for t in trades]
    pairs = [(s, o) for s, o in zip(scores, outcomes) if s is not None]
    if not pairs:
        print(f"  (no scored trades)")
        return
    s = [p for p, _ in pairs]
    o = [q for _, q in pairs]

    bs = brier_score(s, o)
    ref = reference_brier(o)
    sk = skill_score(s, o)
    ece = expected_calibration_error(s, o)
    ll = log_loss(s, o)
    pnl = sum(t.get("pnl_usd") or 0 for t in trades)
    uncensored = sum(1 for t in trades if not t.get("tb_censored", True))

    print(f"  trades          : {len(trades)}  ({uncensored} uncensored)")
    print(f"  realised P&L    : ${pnl:+.2f}")
    print(f"  Brier (raw)     : {bs:.4f}   [base-rate reference: {ref:.4f}]")
    print(f"  skill score     : {sk:+.4f}   {'BEATS base rate' if sk and sk > 0 else 'WORSE than base rate'}")
    print(f"  calibration err : {ece:.4f}")
    print(f"  log loss        : {ll:.4f}")

    curve = reliability_curve(s, o, n_bins=5)
    if curve:
        print("  reliability:")
        print(f"    {'bin':>12} {'n':>4} {'stated':>8} {'actual':>8} {'gap':>7}")
        for row in curve:
            print(
                f"    {row['bin_low']:.1f}-{row['bin_high']:.1f} "
                f"{row['n']:>8} {row['mean_forecast']:>8.3f} "
                f"{row['observed_freq']:>8.3f} {row['gap']:>+7.3f}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, don't write models")
    ap.add_argument("--min-labels", type=int, default=100,
                    help="uncensored labels required before fitting the secondary model")
    ap.add_argument("--config", default="config/config.yaml",
                    help="Read the resolved-trades path from this config, so a "
                         "variant's own history is refit rather than the default's.")
    args = ap.parse_args()

    _apply_config_paths(args.config)
    trades = load_resolved()
    if not trades:
        print("No resolved trades found.")
        return 1

    print("=" * 72)
    print("CALIBRATION & MODEL REFIT")
    print("=" * 72)

    by_variant: Dict[str, List[Dict]] = defaultdict(list)
    for t in trades:
        by_variant[t.get("strategy_variant") or "untagged"].append(t)

    for variant in sorted(by_variant):
        print(f"\n── variant: {variant} " + "─" * (52 - len(variant)))
        report_variant(variant, by_variant[variant])

    print("\n" + "=" * 72)
    print("REFIT")
    print("=" * 72)

    for variant, vtrades in sorted(by_variant.items()):
        scores = [t.get("conviction") for t in vtrades]
        outcomes = [outcome_of(t) for t in vtrades]

        cal = PlattCalibrator().fit(scores, outcomes)
        if not cal.fitted:
            print(f"\n{variant}: calibrator NOT fitted "
                  f"({cal.n_samples} samples, need {PlattCalibrator.MIN_SAMPLES}) "
                  f"— raw score passes through unchanged")
        elif not cal.improves_on_raw():
            print(f"\n{variant}: calibrator fitted but does NOT improve Brier "
                  f"({cal.fit_metrics['brier_raw']:.4f} -> "
                  f"{cal.fit_metrics['brier_calibrated']:.4f}) — not saved")
        else:
            print(f"\n{variant}: calibrator fitted on {cal.n_samples} samples")
            print(f"  Brier {cal.fit_metrics['brier_raw']:.4f} -> "
                  f"{cal.fit_metrics['brier_calibrated']:.4f}")
            print(f"  ECE   {cal.fit_metrics['ece_raw']:.4f} -> "
                  f"{cal.fit_metrics['ece_calibrated']:.4f}")
            print(f"  skill after calibration: {cal.fit_metrics['skill_calibrated']:+.4f}")
            print(f"  method advice: {cal.recommend_method(cal.n_samples)}")
            if not args.dry_run:
                cal.save(variant=variant)
                # Pre-variant history is the general prior for any new variant, so
                # it also lands under "default" — that's the key a freshly-tagged
                # variant falls back to before it has a history of its own.
                if variant == "untagged":
                    cal.save(variant="default")
                    print("  saved (as 'untagged' and 'default').")
                else:
                    print("  saved.")

        # Adaptive segment tracker — rebuilt from full history so it reflects
        # everything, not only what accumulated since it was switched on.
        tracker = SegmentTracker.rebuild_from_history(
            vtrades,
            GATING_DIMENSIONS + DIAGNOSTIC_DIMENSIONS,
            outcome_fn=lambda t: 1 if (t.get("pnl_pct") or 0) > 0 else 0,
            half_life_days=90.0,
            min_samples=8.0,
        )
        rows = tracker.report(break_even=0.5)
        if rows:
            print(f"\n  segment performance (worst first):")
            print(f"    {'segment':<26}{'n_eff':>7}{'win%':>7}{'CI hi':>7}{'P&L':>9}  status")
            for r in rows[:14]:
                gate = "BLOCKED" if r["suppressed"] else ""
                leaky = any(r["segment"].startswith(d + "=") for d in DIAGNOSTIC_DIMENSIONS)
                note = "  (diagnostic only)" if leaky else ""
                print(f"    {r['segment']:<26}{r['n_effective']:>7.1f}"
                      f"{r['win_rate']*100:>6.0f}%{r['ci_high']*100:>6.0f}%"
                      f"{r['total_pnl']:>9.2f}  {gate}{note}")
        if not args.dry_run:
            # Persist only the gating dimensions; diagnostics must never gate.
            gate_tracker = SegmentTracker.rebuild_from_history(
                vtrades, GATING_DIMENSIONS,
                outcome_fn=lambda t: 1 if (t.get("pnl_pct") or 0) > 0 else 0,
                half_life_days=90.0, min_samples=8.0,
            )
            gate_tracker.save(variant=variant)
            if variant == "untagged":
                gate_tracker.save(variant="default")
            print("  segment tracker saved.")

        meta = MetaLabeler(min_training_labels=args.min_labels).fit(vtrades)
        if meta.fitted:
            print(f"  secondary model fitted on {meta.n_samples} labels, "
                  f"AUC {meta.fit_metrics['auc']:.3f}")
            if not args.dry_run:
                meta.save(variant=variant)
                print("  saved.")
        else:
            print(f"  secondary model NOT fitted "
                  f"({meta.n_samples} uncensored labels, need {args.min_labels}) "
                  f"— LLM remains the secondary model")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
