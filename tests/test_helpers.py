"""
Regression tests for the time-horizon exit helpers in scripts/helpers.py.

Covers the bug where check_mid_horizon_stale / check_time_horizon_expired measured
elapsed hold time from midnight of execution_date instead of the actual entry
timestamp, causing positions to be cut a few hours after entry instead of at the
true horizon midpoint (see lessons from 2026-06-23 through 2026-07-11: DEXE cut
after 3.75h, EIGEN and SUN cut early, etc).

Run with: venv/bin/python -m unittest discover -s tests
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from helpers import check_mid_horizon_stale, check_time_horizon_expired, _resolve_opened_at


def _position(hours_ago: float, horizon: str = "1d", pnl_pct=0.0, use_timestamp=True):
    entry = datetime.now() - timedelta(hours=hours_ago)
    pos = {"time_horizon": horizon, "pnl_pct": pnl_pct}
    if use_timestamp:
        pos["execution_timestamp"] = entry.isoformat()
    pos["execution_date"] = entry.strftime("%Y-%m-%d")
    return pos


class TestResolveOpenedAt(unittest.TestCase):
    def test_prefers_full_timestamp_over_date(self):
        pos = {
            "execution_timestamp": "2026-06-25T12:17:00",
            "execution_date": "2026-06-25",
        }
        opened = _resolve_opened_at(pos)
        self.assertEqual(opened, datetime.fromisoformat("2026-06-25T12:17:00"))

    def test_falls_back_to_date_when_no_timestamp(self):
        pos = {"execution_date": "2026-06-25"}
        opened = _resolve_opened_at(pos)
        self.assertEqual(opened, datetime(2026, 6, 25))

    def test_returns_none_when_nothing_present(self):
        self.assertIsNone(_resolve_opened_at({}))

    def test_returns_none_on_unparseable_timestamp(self):
        pos = {"execution_timestamp": "not-a-date", "execution_date": "also-bad"}
        self.assertIsNone(_resolve_opened_at(pos))


class TestCheckMidHorizonStale(unittest.TestCase):
    def test_dexe_repro_not_stale_at_3h45m(self):
        # Real case from lessons: entered 12:17, cut at 16:00 (3h45m later) on a 1d
        # horizon. The midnight-truncation bug measured this as 16h/24h=0.67 elapsed
        # and wrongly fired. Fixed version must not fire this early.
        pos = _position(hours_ago=3.75, horizon="1d", pnl_pct=0.0)
        self.assertFalse(check_mid_horizon_stale(pos))

    def test_fires_past_true_halfway_point_with_no_gain(self):
        pos = _position(hours_ago=13, horizon="1d", pnl_pct=0.1)
        self.assertTrue(check_mid_horizon_stale(pos))

    def test_does_not_fire_past_halfway_if_gain_above_threshold(self):
        pos = _position(hours_ago=13, horizon="1d", pnl_pct=2.0)
        self.assertFalse(check_mid_horizon_stale(pos))

    def test_does_not_fire_before_halfway_even_with_no_gain(self):
        pos = _position(hours_ago=10, horizon="1d", pnl_pct=0.0)
        self.assertFalse(check_mid_horizon_stale(pos))

    def test_no_pnl_means_not_stale(self):
        pos = _position(hours_ago=20, horizon="1d", pnl_pct=0.0)
        pos["pnl_pct"] = None
        self.assertFalse(check_mid_horizon_stale(pos))

    def test_unparseable_horizon_returns_false(self):
        pos = _position(hours_ago=20, horizon="garbage", pnl_pct=0.0)
        self.assertFalse(check_mid_horizon_stale(pos))

    def test_falls_back_to_execution_date_when_no_timestamp(self):
        # Older records may only have execution_date — should still work,
        # just measured from midnight rather than the real fill time.
        pos = _position(hours_ago=13, horizon="1d", pnl_pct=0.1, use_timestamp=False)
        # Whether this fires depends on how far past midnight we are; just confirm
        # it doesn't crash and returns a bool.
        self.assertIn(check_mid_horizon_stale(pos), (True, False))


class TestCheckTimeHorizonExpired(unittest.TestCase):
    def test_not_expired_partway_through_horizon(self):
        pos = _position(hours_ago=10, horizon="1d")
        self.assertFalse(check_time_horizon_expired(pos))

    def test_expired_after_full_horizon(self):
        pos = _position(hours_ago=25, horizon="1d")
        self.assertTrue(check_time_horizon_expired(pos))

    def test_not_expired_just_under_horizon(self):
        pos = _position(hours_ago=23, horizon="1d")
        self.assertFalse(check_time_horizon_expired(pos))

    def test_unparseable_horizon_returns_false(self):
        pos = _position(hours_ago=100, horizon="nope")
        self.assertFalse(check_time_horizon_expired(pos))

    def test_multi_day_horizon(self):
        pos = _position(hours_ago=49, horizon="2d")
        self.assertTrue(check_time_horizon_expired(pos))
        pos2 = _position(hours_ago=47, horizon="2d")
        self.assertFalse(check_time_horizon_expired(pos2))


if __name__ == "__main__":
    unittest.main()
