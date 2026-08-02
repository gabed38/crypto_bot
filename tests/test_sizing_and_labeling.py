"""
Tests for volatility-targeted sizing, triple-barrier labeling, and the
meta-labeling secondary model.

Run: venv/bin/python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.labeling import compute_barriers, is_trainable, label_closed_trade
from src.analysis.meta_labeler import MetaLabeler
from src.trading.sizing import size_crypto_trade, vol_target_size


class TestVolTargetSizing(unittest.TestCase):
    def test_coin_at_target_vol_gets_base_size(self):
        self.assertAlmostEqual(vol_target_size(5.0, daily_vol_pct=3.0, target_vol_pct=3.0), 5.0)

    def test_high_vol_coin_gets_smaller_position(self):
        # 6%/day is twice the 3% target -> half size
        self.assertAlmostEqual(vol_target_size(5.0, daily_vol_pct=6.0, target_vol_pct=3.0), 2.5)

    def test_low_vol_coin_gets_larger_position(self):
        self.assertAlmostEqual(vol_target_size(5.0, daily_vol_pct=1.5, target_vol_pct=3.0), 10.0)

    def test_equal_risk_across_very_different_coins(self):
        # The whole point: size x vol should be constant
        a = vol_target_size(5.0, 2.0, target_vol_pct=3.0, max_usd=1e9)
        b = vol_target_size(5.0, 12.0, target_vol_pct=3.0, max_usd=1e9)
        self.assertAlmostEqual(a * 2.0, b * 12.0, places=4)

    def test_max_cap_respected(self):
        self.assertEqual(vol_target_size(5.0, 0.5, target_vol_pct=3.0, max_usd=15.0), 15.0)

    def test_min_floor_respected(self):
        self.assertEqual(vol_target_size(5.0, 100.0, target_vol_pct=3.0, min_usd=1.0), 1.0)

    def test_unknown_vol_falls_back_to_base_not_upsized(self):
        # Missing data must never silently produce a large position
        self.assertEqual(vol_target_size(5.0, None), 5.0)
        self.assertEqual(vol_target_size(5.0, "n/a"), 5.0)

    def test_zero_or_negative_vol_falls_back(self):
        self.assertEqual(vol_target_size(5.0, 0.0), 5.0)
        self.assertEqual(vol_target_size(5.0, -2.0), 5.0)

    def test_size_crypto_trade_flat_mode_ignores_vol(self):
        out = size_crypto_trade({"daily_vol_pct": 20.0},
                                {"sizing_method": "flat", "base_position_usd": 5.0})
        self.assertEqual(out["amount_invested"], 5.0)
        self.assertEqual(out["sizing_method"], "flat")

    def test_size_crypto_trade_records_audit_trail(self):
        out = size_crypto_trade(
            {"daily_vol_pct": 6.0},
            {"sizing_method": "vol_target", "base_position_usd": 5.0,
             "target_vol_pct": 3.0, "max_position_size_usd": 15.0},
        )
        self.assertEqual(out["amount_invested"], 2.5)
        self.assertEqual(out["sizing_daily_vol_pct"], 6.0)
        self.assertAlmostEqual(out["sizing_scale_factor"], 0.5)


class TestTripleBarrier(unittest.TestCase):
    def test_barriers_scale_with_volatility(self):
        low = compute_barriers(2.0, profit_mult=2.0, stop_mult=2.0)
        high = compute_barriers(8.0, profit_mult=2.0, stop_mult=2.0)
        self.assertLess(low["profit_barrier_pct"], high["profit_barrier_pct"])
        self.assertEqual(low["barrier_basis"], "volatility_scaled")

    def test_barriers_clamped_to_sane_range(self):
        extreme = compute_barriers(90.0)
        self.assertLessEqual(extreme["profit_barrier_pct"], 25.0)
        tiny = compute_barriers(0.1)
        self.assertGreaterEqual(tiny["profit_barrier_pct"], 3.0)

    def test_unknown_vol_uses_fixed_fallback(self):
        out = compute_barriers(None)
        self.assertEqual(out["barrier_basis"], "fallback_fixed")

    def test_profit_exits_label_positive(self):
        for ct in ("TAKE_PROFIT", "PROFIT_PROTECTION"):
            self.assertEqual(label_closed_trade({"close_type": ct})["tb_label"], 1)

    def test_stop_exits_label_negative(self):
        for ct in ("STOP_LOSS", "TRAILING_STOP", "CUT_LOSS"):
            self.assertEqual(label_closed_trade({"close_type": ct})["tb_label"], -1)

    def test_time_expiry_labels_neutral_and_is_trainable(self):
        lab = label_closed_trade({"close_type": "TIME_EXPIRED"})
        self.assertEqual(lab["tb_label"], 0)
        self.assertFalse(lab["tb_censored"])
        self.assertTrue(is_trainable(lab))

    def test_discretionary_exits_are_censored(self):
        # The premature-exit bug produced 20 of these; they must not train the model
        for ct in ("STALE_POSITION", "STRATEGY_RESET"):
            lab = label_closed_trade({"close_type": ct})
            self.assertTrue(lab["tb_censored"])
            self.assertFalse(is_trainable(lab))

    def test_unknown_close_type_is_censored_not_assumed(self):
        lab = label_closed_trade({"close_type": "SOMETHING_NEW"})
        self.assertTrue(lab["tb_censored"])

    def test_label_is_independent_of_pnl_sign(self):
        # A profit-barrier hit is +1 even if fees made P&L look flat
        lab = label_closed_trade({"close_type": "TAKE_PROFIT", "pnl_pct": 0.0})
        self.assertEqual(lab["tb_label"], 1)


class TestMetaLabeler(unittest.TestCase):
    def _rows(self, n, censored=False):
        rows = []
        for i in range(n):
            rows.append({
                "screen_score": float(i % 20),
                "daily_vol_pct": 3.0 + (i % 5),
                "stop_multiple": 2.0,
                "price_change_24h": -5.0 if i % 2 else 5.0,
                "rsi": 30 + (i % 40),
                "market_cap": 1e9,
                "volume_24h": 1e8,
                "tb_label": 1 if i % 2 else -1,
                "tb_censored": censored,
            })
        return rows

    def test_refuses_to_fit_below_threshold(self):
        m = MetaLabeler(min_training_labels=100).fit(self._rows(30))
        self.assertFalse(m.fitted)
        self.assertEqual(m.backend, "llm")

    def test_fits_above_threshold(self):
        m = MetaLabeler(min_training_labels=50).fit(self._rows(120))
        self.assertTrue(m.fitted)
        self.assertEqual(m.backend, "classifier")

    def test_censored_rows_excluded_from_training(self):
        m = MetaLabeler(min_training_labels=10).fit(self._rows(100, censored=True))
        self.assertEqual(m.n_samples, 0)
        self.assertFalse(m.fitted)

    def test_llm_mode_defers_to_conviction(self):
        m = MetaLabeler(min_training_labels=1000)
        take = m.decide({"conviction": 0.8}, threshold=0.7)
        skip = m.decide({"conviction": 0.5}, threshold=0.7)
        self.assertEqual(take["meta_decision"], "take")
        self.assertEqual(take["meta_source"], "llm")
        self.assertEqual(skip["meta_decision"], "skip")

    def test_classifier_mode_ignores_conviction(self):
        m = MetaLabeler(min_training_labels=50).fit(self._rows(120))
        d = m.decide({"conviction": 0.99, "screen_score": 1.0, "daily_vol_pct": 3.0,
                      "market_cap": 1e9, "volume_24h": 1e8}, threshold=0.5)
        self.assertEqual(d["meta_source"], "classifier")
        self.assertNotAlmostEqual(d["meta_confidence"], 0.99)

    def test_predictions_are_probabilities(self):
        m = MetaLabeler(min_training_labels=50).fit(self._rows(120))
        for score in (0.0, 5.0, 50.0, -50.0):
            p = m.predict_proba({"screen_score": score, "market_cap": 1e9, "volume_24h": 1e8})
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_missing_features_do_not_crash(self):
        m = MetaLabeler(min_training_labels=50).fit(self._rows(120))
        self.assertIsInstance(m.predict_proba({}), float)

    def test_roundtrip_serialisation(self):
        m = MetaLabeler(min_training_labels=50).fit(self._rows(120))
        restored = MetaLabeler.load.__func__(MetaLabeler, Path("/nonexistent.json"))
        self.assertFalse(restored.fitted)  # missing file -> safe default
        d = m.to_dict()
        self.assertIn("weights", d)
        self.assertTrue(d["fitted"])


if __name__ == "__main__":
    unittest.main()
