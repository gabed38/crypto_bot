"""
Tests for probability calibration and proper scoring.

Run: venv/bin/python -m unittest discover -s tests
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.calibration import (
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    log_loss,
    reference_brier,
    reliability_curve,
    skill_score,
)


class TestScoringRules(unittest.TestCase):
    def test_brier_perfect_forecast_is_zero(self):
        self.assertEqual(brier_score([1.0, 0.0, 1.0], [1, 0, 1]), 0.0)

    def test_brier_maximally_wrong_is_one(self):
        self.assertEqual(brier_score([0.0, 1.0], [1, 0]), 1.0)

    def test_brier_always_half_is_quarter(self):
        self.assertAlmostEqual(brier_score([0.5] * 4, [1, 0, 1, 0]), 0.25)

    def test_brier_none_on_empty(self):
        self.assertIsNone(brier_score([], []))

    def test_reference_brier_uses_base_rate(self):
        # base rate 0.5 -> 0.25
        self.assertAlmostEqual(reference_brier([1, 0, 1, 0]), 0.25)

    def test_skill_positive_when_better_than_base_rate(self):
        # Confident and correct beats guessing the base rate
        probs = [0.9, 0.1, 0.9, 0.1]
        outs = [1, 0, 1, 0]
        self.assertGreater(skill_score(probs, outs), 0)

    def test_skill_negative_when_worse_than_base_rate(self):
        # Confident and wrong
        probs = [0.9, 0.9, 0.1, 0.1]
        outs = [0, 0, 1, 1]
        self.assertLess(skill_score(probs, outs), 0)

    def test_log_loss_punishes_confident_error(self):
        mild = log_loss([0.6], [0])
        severe = log_loss([0.99], [0])
        self.assertGreater(severe, mild)

    def test_log_loss_does_not_overflow_on_certainty(self):
        # p=0 with outcome 1 must be clamped, not infinite
        self.assertTrue(log_loss([0.0], [1]) < float("inf"))

    def test_scoring_ignores_missing_rows(self):
        self.assertAlmostEqual(brier_score([0.5, None, 0.5], [1, 1, 0]), 0.25)


class TestReliabilityCurve(unittest.TestCase):
    def test_perfectly_calibrated_has_no_gap(self):
        # 100 forecasts at 0.5, exactly half true
        probs = [0.5] * 100
        outs = [1] * 50 + [0] * 50
        curve = reliability_curve(probs, outs, n_bins=10)
        self.assertEqual(len(curve), 1)
        self.assertAlmostEqual(curve[0]["gap"], 0.0, places=6)

    def test_overconfident_model_shows_negative_gap(self):
        # Claims 0.9, delivers 0.5 — the signature found in both bots
        probs = [0.9] * 100
        outs = [1] * 50 + [0] * 50
        curve = reliability_curve(probs, outs, n_bins=10)
        self.assertLess(curve[0]["gap"], -0.3)

    def test_ece_zero_when_calibrated(self):
        probs = [0.5] * 100
        outs = [1] * 50 + [0] * 50
        self.assertAlmostEqual(expected_calibration_error(probs, outs), 0.0, places=6)

    def test_empty_input_returns_empty_curve(self):
        self.assertEqual(reliability_curve([], []), [])


class TestPlattCalibrator(unittest.TestCase):
    def test_identity_until_fitted(self):
        c = PlattCalibrator()
        self.assertFalse(c.fitted)
        self.assertAlmostEqual(c.transform(0.73), 0.73)

    def test_refuses_to_fit_below_min_samples(self):
        c = PlattCalibrator().fit([0.7] * 5, [1, 0, 1, 0, 1])
        self.assertFalse(c.fitted)
        self.assertAlmostEqual(c.transform(0.7), 0.7)

    def test_fixes_systematic_overconfidence(self):
        # Scores cluster 0.7-0.9 but the true rate is ~50% — the real pattern.
        random.seed(7)
        scores, outcomes = [], []
        for _ in range(400):
            s = random.uniform(0.7, 0.9)
            scores.append(s)
            outcomes.append(1 if random.random() < 0.5 else 0)
        c = PlattCalibrator().fit(scores, outcomes)
        self.assertTrue(c.fitted)
        # Calibrated output should sit near the true base rate, not near 0.8
        self.assertLess(abs(c.transform(0.8) - 0.5), 0.12)

    def test_fit_improves_brier_on_miscalibrated_data(self):
        random.seed(11)
        scores, outcomes = [], []
        for _ in range(400):
            s = random.uniform(0.7, 0.9)
            scores.append(s)
            outcomes.append(1 if random.random() < 0.5 else 0)
        c = PlattCalibrator().fit(scores, outcomes)
        self.assertLess(c.fit_metrics["brier_calibrated"], c.fit_metrics["brier_raw"])
        self.assertTrue(c.improves_on_raw())

    def test_preserves_real_signal(self):
        # When the score genuinely predicts, calibration must not flatten it
        random.seed(3)
        scores, outcomes = [], []
        for _ in range(400):
            s = random.uniform(0.0, 1.0)
            scores.append(s)
            outcomes.append(1 if random.random() < s else 0)
        c = PlattCalibrator().fit(scores, outcomes)
        self.assertGreater(c.transform(0.8), c.transform(0.3))

    def test_output_always_a_probability(self):
        c = PlattCalibrator(a=50.0, b=-20.0, fitted=True)
        for s in (-5.0, 0.0, 0.5, 1.0, 99.0):
            p = c.transform(s)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_none_score_returns_neutral(self):
        self.assertEqual(PlattCalibrator().transform(None), 0.5)

    def test_roundtrip_serialisation(self):
        c = PlattCalibrator(a=2.5, b=-1.25, fitted=True)
        c.n_samples = 150
        restored = PlattCalibrator.from_dict(c.to_dict())
        self.assertAlmostEqual(restored.a, 2.5)
        self.assertAlmostEqual(restored.b, -1.25)
        self.assertTrue(restored.fitted)
        self.assertAlmostEqual(restored.transform(0.7), c.transform(0.7))

    def test_method_recommendation_scales_with_sample_size(self):
        self.assertIn("none", PlattCalibrator.recommend_method(10))
        self.assertIn("platt", PlattCalibrator.recommend_method(150))
        self.assertIn("isotonic", PlattCalibrator.recommend_method(5000))


if __name__ == "__main__":
    unittest.main()
