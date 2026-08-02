"""
Tests for adaptive segment learning.

The behaviour that matters most here is RESTRAINT: a system that reacts to a
three-loss streak will chase noise and thrash. Most of these tests assert that
the tracker does nothing when the evidence is thin.

Run: venv/bin/python -m unittest discover -s tests
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.adaptive import (
    DIAGNOSTIC_DIMENSIONS,
    GATING_DIMENSIONS,
    SegmentTracker,
    _wilson_interval,
)

DIMS = ("sector", "vol_signal")


def _tracker(min_samples=8.0, half_life=30.0):
    return SegmentTracker(half_life_days=half_life, min_samples=min_samples)


def _feed(tracker, sector, wins, losses, days_ago=0):
    when = datetime.now() - timedelta(days=days_ago)
    for _ in range(wins):
        tracker.record({"sector": sector}, DIMS, won=True, pnl=1.0, when=when)
    for _ in range(losses):
        tracker.record({"sector": sector}, DIMS, won=False, pnl=-1.0, when=when)


class TestWilsonInterval(unittest.TestCase):
    def test_bounds_stay_within_zero_one(self):
        for p, n in ((0.0, 3), (1.0, 3), (0.5, 1), (0.02, 50)):
            lo, hi = _wilson_interval(p, n, 1.28)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_interval_narrows_with_more_data(self):
        _, hi_small = _wilson_interval(0.5, 5, 1.28)
        _, hi_large = _wilson_interval(0.5, 500, 1.28)
        self.assertLess(hi_large, hi_small)

    def test_zero_n_is_maximally_uncertain(self):
        self.assertEqual(_wilson_interval(0.5, 0, 1.28), (0.0, 1.0))


class TestRestraint(unittest.TestCase):
    """The tracker must not act on thin evidence."""

    def test_three_straight_losses_does_not_suppress(self):
        t = _tracker()
        _feed(t, "DeFi", wins=0, losses=3)
        suppress, reason = t.should_suppress("sector=DeFi")
        self.assertFalse(suppress)
        self.assertIn("insufficient evidence", reason)

    def test_zero_percent_win_rate_on_one_trade_does_not_suppress(self):
        t = _tracker()
        _feed(t, "Privacy", wins=0, losses=1)
        self.assertFalse(t.should_suppress("sector=Privacy")[0])

    def test_unknown_segment_does_not_suppress(self):
        self.assertFalse(_tracker().should_suppress("sector=NeverSeen")[0])

    def test_mediocre_but_not_damning_record_survives(self):
        # 45% over 20 trades — bad, but the upper bound still reaches break-even
        t = _tracker()
        _feed(t, "AI", wins=9, losses=11)
        self.assertFalse(t.should_suppress("sector=AI")[0])


class TestSuppression(unittest.TestCase):
    """It must act when the evidence really is damning."""

    def test_clearly_losing_segment_is_suppressed(self):
        t = _tracker()
        _feed(t, "Meme", wins=2, losses=28)
        suppress, reason = t.should_suppress("sector=Meme")
        self.assertTrue(suppress)
        self.assertIn("break-even", reason)

    def test_suppression_respects_break_even_parameter(self):
        # 55% wins: fine at break-even 0.5, not fine at 0.8
        t = _tracker()
        _feed(t, "L1", wins=33, losses=27)
        self.assertFalse(t.should_suppress("sector=L1", break_even=0.5)[0])
        self.assertTrue(t.should_suppress("sector=L1", break_even=0.8)[0])

    def test_winning_segment_never_suppressed(self):
        t = _tracker()
        _feed(t, "Infra", wins=28, losses=2)
        self.assertFalse(t.should_suppress("sector=Infra")[0])


class TestRecencyDecay(unittest.TestCase):
    def test_old_evidence_stops_suppressing(self):
        # Damning record, but a year old with a 30-day half-life
        t = _tracker(half_life=30.0)
        _feed(t, "Gaming", wins=1, losses=29, days_ago=365)
        self.assertFalse(t.should_suppress("sector=Gaming")[0])

    def test_recent_evidence_outweighs_old(self):
        t = _tracker(half_life=30.0)
        _feed(t, "Mixed", wins=0, losses=20, days_ago=400)   # ancient, should fade
        _feed(t, "Mixed", wins=18, losses=2, days_ago=0)      # fresh, should dominate
        s = t.stats("sector=Mixed")
        self.assertGreater(s["win_rate"], 0.8)

    def test_effective_count_below_raw_count_when_aged(self):
        t = _tracker(half_life=30.0)
        _feed(t, "Aged", wins=5, losses=5, days_ago=60)
        s = t.stats("sector=Aged")
        self.assertEqual(s["n_raw"], 10)
        self.assertLess(s["n_effective"], 10)


class TestLeakageGuard(unittest.TestCase):
    def test_gating_on_post_hoc_dimension_raises(self):
        t = _tracker()
        with self.assertRaises(ValueError) as ctx:
            t.evaluate_trade({"close_type": "STOP_LOSS"}, ("close_type",))
        self.assertIn("not knowable at entry", str(ctx.exception))

    def test_gating_dimensions_are_disjoint_from_diagnostic(self):
        self.assertEqual(set(GATING_DIMENSIONS) & set(DIAGNOSTIC_DIMENSIONS), set())

    def test_default_gating_dimensions_are_safe(self):
        # Should not raise
        _tracker().evaluate_trade({"sector": "DeFi"})


class TestEvaluateTrade(unittest.TestCase):
    def test_blocked_when_any_segment_is_bad(self):
        t = _tracker()
        _feed(t, "Meme", wins=1, losses=29)
        out = t.evaluate_trade({"sector": "Meme", "vol_signal": "LOW"}, DIMS)
        self.assertTrue(out["adaptive_blocked"])
        self.assertEqual(out["adaptive_blocked_by"][0]["segment"], "sector=Meme")

    def test_clean_trade_passes(self):
        t = _tracker()
        _feed(t, "Infra", wins=20, losses=10)
        out = t.evaluate_trade({"sector": "Infra", "vol_signal": "MEDIUM"}, DIMS)
        self.assertFalse(out["adaptive_blocked"])
        self.assertEqual(out["adaptive_blocked_by"], [])

    def test_missing_dimensions_are_skipped_not_crashed(self):
        out = _tracker().evaluate_trade({}, DIMS)
        self.assertFalse(out["adaptive_blocked"])


class TestPersistence(unittest.TestCase):
    def test_roundtrip_preserves_statistics(self):
        t = _tracker()
        _feed(t, "DeFi", wins=6, losses=4)
        restored = SegmentTracker.load.__func__(SegmentTracker, Path("/nonexistent.json"))
        self.assertEqual(restored.observations, {})  # missing file -> empty, not crash

        d = t.to_dict()
        self.assertIn("sector=DeFi", d["observations"])
        self.assertEqual(len(d["observations"]["sector=DeFi"]), 10)

    def test_rebuild_from_history(self):
        trades = [
            {"sector": "DeFi", "pnl_usd": 1.0, "resolved_at": datetime.now().isoformat()},
            {"sector": "DeFi", "pnl_usd": -1.0, "resolved_at": datetime.now().isoformat()},
        ]
        t = SegmentTracker.rebuild_from_history(
            trades, DIMS, outcome_fn=lambda x: 1 if (x.get("pnl_usd") or 0) > 0 else 0
        )
        s = t.stats("sector=DeFi")
        self.assertEqual(s["n_raw"], 2)
        self.assertAlmostEqual(s["win_rate"], 0.5, places=2)


class TestReport(unittest.TestCase):
    def test_report_orders_suppressed_first(self):
        t = _tracker()
        _feed(t, "Bad", wins=1, losses=29)
        _feed(t, "Good", wins=25, losses=5)
        rows = t.report()
        self.assertTrue(rows[0]["suppressed"])
        self.assertEqual(rows[0]["segment"], "sector=Bad")


if __name__ == "__main__":
    unittest.main()
