"""Risk management for crypto trading bot."""

from typing import Dict, Any, List, Tuple
from loguru import logger


class RiskManager:
    """Enforce trading risk limits."""

    def __init__(
        self,
        max_daily_loss_pct_of_committed: float = 0.25,
        max_daily_loss_floor_usd: float = 5.0,
        max_position_percentage: float = 0.2,
        stop_loss_percentage: float = 0.20,
        drawdown_warning_pct: float = -15.0,
        drawdown_halt_pct: float = -25.0,
    ):
        """
        Args:
            max_daily_loss_pct_of_committed: Max daily realized loss as a fraction of
                total committed capital. E.g. 0.25 = 25% of what's currently invested.
                With $5/trade and 10 open positions ($50 committed), the limit is $12.50.
            max_daily_loss_floor_usd: Minimum daily loss limit regardless of committed
                capital. Prevents the limit from being near-zero when few positions are open.
                Default $5 = one full position.
            stop_loss_percentage: Fallback stop-loss fraction (default 20%) used only
                when a position has no ``adaptive_stop_pct`` stamped at entry.
            drawdown_warning_pct: Unrealized portfolio drawdown % that triggers a warning
                and more selective trading. Negative number (e.g. -15.0 = -15%).
            drawdown_halt_pct: Unrealized portfolio drawdown % that halts new trades
                entirely until recovery. Negative number (e.g. -25.0 = -25%).
        """
        self.max_daily_loss_pct_of_committed = max_daily_loss_pct_of_committed
        self.max_daily_loss_floor_usd = max_daily_loss_floor_usd
        self.max_position_percentage = max_position_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.drawdown_warning_pct = drawdown_warning_pct
        self.drawdown_halt_pct = drawdown_halt_pct

    # ── Dynamic daily loss limit ──────────────────────────────────────────────

    def calculate_dynamic_daily_limit(
        self,
        open_positions: List[Dict[str, Any]],
        max_position_size_usd: float,
    ) -> float:
        """
        Compute the daily realized loss limit based on committed capital.

        Formula:
            committed = len(open_positions) * max_position_size_usd
            limit = max(committed * pct_of_committed, floor)

        Examples (with default 25% and $5 floor):
            0 positions  ($0 committed)  → $5.00  (floor)
            4 positions  ($20 committed) → $5.00  (max of $5, floor)
            10 positions ($50 committed) → $12.50
            20 positions ($100 committed)→ $25.00
        """
        committed = len(open_positions) * float(max_position_size_usd)
        limit = committed * self.max_daily_loss_pct_of_committed
        return max(limit, self.max_daily_loss_floor_usd)

    # ── Drawdown circuit breaker ──────────────────────────────────────────────

    def check_portfolio_drawdown(
        self,
        open_positions: List[Dict[str, Any]],
    ) -> Tuple[str, float, str]:
        """
        Evaluate unrealised portfolio drawdown across all open positions.

        Returns:
            (status, drawdown_pct, message)
            status: "ok" | "warn" | "halt"
            drawdown_pct: current drawdown as a percentage (negative = loss)
            message: human-readable explanation

        Thresholds:
            ok   → drawdown_pct > warning_pct
            warn → drawdown_halt_pct < drawdown_pct <= warning_pct
            halt → drawdown_pct <= drawdown_halt_pct
        """
        if not open_positions:
            return "ok", 0.0, "No open positions"

        total_invested = sum(
            float(p.get("amount_invested") or 5.0) for p in open_positions
        )
        if total_invested == 0:
            return "ok", 0.0, "No invested capital"

        total_unrealised_pnl = sum(
            float(p.get("pnl_usd") or 0) for p in open_positions
        )
        drawdown_pct = (total_unrealised_pnl / total_invested) * 100

        if drawdown_pct <= self.drawdown_halt_pct:
            msg = (
                f"Portfolio drawdown {drawdown_pct:.1f}% exceeds halt threshold "
                f"({self.drawdown_halt_pct:.1f}%). Blocking new trades until recovery."
            )
            logger.warning(msg)
            return "halt", drawdown_pct, msg

        if drawdown_pct <= self.drawdown_warning_pct:
            msg = (
                f"Portfolio drawdown {drawdown_pct:.1f}% — approaching halt threshold "
                f"({self.drawdown_halt_pct:.1f}%). Be more selective this run."
            )
            logger.warning(msg)
            return "warn", drawdown_pct, msg

        return "ok", drawdown_pct, f"Portfolio drawdown {drawdown_pct:.1f}% within limits"

    # ── Trade limit check ─────────────────────────────────────────────────────

    def check_trade_limits(
        self,
        proposed_trades: List[Dict[str, Any]],
        current_pnl: float,
        open_positions: List[Dict[str, Any]],
        max_position_size_usd: float,
        trading_enabled: bool = True,
    ) -> Tuple[bool, str]:
        """
        Check if proposed trades violate risk limits.

        Loss checks are skipped in paper trading mode (trading_enabled=False).

        Args:
            current_pnl: Today's realized P&L so far (from PnLTracker).
            open_positions: Current open positions (used to compute dynamic limit).
            max_position_size_usd: Per-trade size from config.

        Returns:
            (allowed, reason) tuple
        """
        if trading_enabled:
            daily_limit = self.calculate_dynamic_daily_limit(
                open_positions, max_position_size_usd
            )

            if current_pnl < -daily_limit:
                return (
                    False,
                    f"Daily loss limit hit (${current_pnl:.2f} realized today / "
                    f"-${daily_limit:.2f} limit based on ${len(open_positions) * max_position_size_usd:.2f} committed)",
                )

            potential_loss = sum(self._estimate_position_value(t) for t in proposed_trades)
            if current_pnl - potential_loss < -daily_limit:
                return (
                    False,
                    f"Proposed trades could breach daily loss limit "
                    f"(-${daily_limit:.2f}). Current P&L: ${current_pnl:.2f}",
                )

        return True, "All risk checks passed"

    def _estimate_position_value(self, trade: Dict[str, Any]) -> float:
        return float(trade.get("amount_invested") or 5.0)

    # ── Trailing stop-loss ────────────────────────────────────────────────────

    def update_trailing_high(
        self,
        position: Dict[str, Any],
        current_price: float,
    ) -> Dict[str, Any]:
        """Update the position's trailing high-water mark and trailing stop price.

        Call once per price observation (every Phase 1 run).  The updated dict
        is a shallow copy of ``position`` with two extra keys:

        - ``highest_price``       — highest observed price for LONGs (lowest for SHORTs)
        - ``trailing_stop_price`` — stop fires if price crosses back through this level

        The trailing stop only becomes active once the position is *in profit*
        (i.e. the trailing stop price is strictly better than the hard stop price).
        Until then, ``check_stop_loss`` handles it.
        """
        updated = dict(position)
        entry = float(position.get("entry_price") or 0)
        direction = position.get("direction", "LONG")
        current_price = float(current_price)

        stop_pct = self._stop_pct_for_position(position)

        if direction == "LONG":
            prev_high = float(position.get("highest_price") or entry)
            new_high = max(prev_high, current_price)
            updated["highest_price"] = new_high
            updated["trailing_stop_price"] = round(
                new_high * (1.0 - stop_pct), 8
            )
        else:
            prev_low = float(position.get("lowest_price") or entry)
            new_low = min(prev_low, current_price)
            updated["lowest_price"] = new_low
            updated["trailing_stop_price"] = round(
                new_low * (1.0 + stop_pct), 8
            )

        return updated

    def check_trailing_stop(
        self,
        position: Dict[str, Any],
        current_price: float,
    ) -> bool:
        """Return True if the trailing stop has been breached.

        The trailing stop is only active once the position has moved into profit
        enough that the trailing stop price is *above* the hard stop baseline.
        This prevents it from firing on new positions that haven't yet gained.

        For LONGs:
            - Hard stop baseline = entry * (1 - stop_pct)
            - Trailing stop fires when current_price <= trailing_stop_price
              AND trailing_stop_price > hard stop baseline
        """
        trailing_stop = position.get("trailing_stop_price")
        if trailing_stop is None:
            return False

        entry = float(position.get("entry_price") or 0)
        direction = position.get("direction", "LONG")
        current_price = float(current_price)
        trailing_stop = float(trailing_stop)
        stop_pct = self._stop_pct_for_position(position)

        if direction == "LONG":
            hard_stop = entry * (1.0 - stop_pct)
            # Only activate once trailing stop has moved above hard stop
            if trailing_stop <= hard_stop:
                return False
            triggered = current_price <= trailing_stop
        else:
            hard_stop = entry * (1.0 + stop_pct)
            if trailing_stop >= hard_stop:
                return False
            triggered = current_price >= trailing_stop

        if triggered:
            logger.warning(
                f"Trailing stop triggered: {position.get('symbol', '?')} "
                f"price={current_price} trailing_stop={trailing_stop:.6f} "
                f"high={position.get('highest_price', position.get('lowest_price', '?'))}"
            )
        return triggered

    # ── Mechanical price-based exit rules ────────────────────────────────────
    # All four rules operate purely on entry price vs current price.
    # No LLM required — exits are deterministic and fast.

    def check_take_profit(
        self,
        position: Dict[str, Any],
        current_price: float,
    ) -> bool:
        """Return True when the position has reached the target set at entry.

        ``target_pct`` is the percentage gain goal stored on the position
        (e.g. 6.0 means close when up 6%).  If no target was set, returns False
        (position stays open until another rule fires).
        """
        entry = float(position.get("entry_price") or 0)
        target_pct = float(position.get("target_pct") or 0)
        if not entry or not target_pct:
            return False

        direction = position.get("direction", "LONG")
        if direction == "LONG":
            gain_pct = (current_price - entry) / entry * 100
        else:
            gain_pct = (entry - current_price) / entry * 100

        if round(gain_pct, 4) >= target_pct:
            logger.info(
                f"Take-profit triggered: {position.get('symbol', '?')} "
                f"+{gain_pct:.1f}% ≥ target {target_pct:.1f}%"
            )
            return True
        return False

    def check_profit_protection(
        self,
        position: Dict[str, Any],
        current_price: float,
        min_peak_pct: float = 5.0,
        giveback_fraction: float = 0.35,
    ) -> bool:
        """Return True when a meaningful gain has reversed significantly.

        Once a position peaks at ``min_peak_pct`` or higher, this rule
        closes it if the current gain falls to ≤ ``giveback_fraction`` × peak.

        Examples (defaults: min_peak=5%, giveback=35%):
            Peak +8%  → close if current ≤ +2.8%
            Peak +6%  → close if current ≤ +2.1%
            Peak +4%  → rule does NOT activate (below min_peak threshold)

        Raised min_peak from 4% to 5%: prevents the rule from firing on tiny
        bounces that barely moved. Lowered giveback from 50% to 35%: when the
        rule does fire, it holds on to more of the peak gain before exiting.

        Peak gain is derived from the already-tracked ``highest_price`` /
        ``lowest_price`` high-water marks, so no extra state is needed.
        """
        entry = float(position.get("entry_price") or 0)
        if not entry:
            return False

        direction = position.get("direction", "LONG")

        if direction == "LONG":
            highest = float(position.get("highest_price") or entry)
            peak_pct = (highest - entry) / entry * 100
            current_pct = (current_price - entry) / entry * 100
        else:
            lowest = float(position.get("lowest_price") or entry)
            peak_pct = (entry - lowest) / entry * 100
            current_pct = (entry - current_price) / entry * 100

        if peak_pct < min_peak_pct:
            return False

        floor = peak_pct * giveback_fraction
        triggered = current_pct <= floor

        if triggered:
            logger.info(
                f"Profit protection triggered: {position.get('symbol', '?')} "
                f"peak={peak_pct:.1f}%  now={current_pct:.1f}%  "
                f"floor={floor:.1f}% (gave back >{giveback_fraction*100:.0f}% of peak)"
            )
        return triggered

    # ── Per-position stop-loss ────────────────────────────────────────────────

    def _stop_pct_for_position(self, position: Dict[str, Any]) -> float:
        """Return the effective stop-loss fraction for this position.

        Uses the vol-adaptive stop stamped at entry (``adaptive_stop_pct``, a
        percentage value such as 12.0) if available, otherwise falls back to the
        bot-wide ``stop_loss_percentage`` (a fraction such as 0.20).
        """
        adaptive = position.get("adaptive_stop_pct")
        if adaptive is not None:
            return float(adaptive) / 100.0
        return self.stop_loss_percentage

    def check_stop_loss(self, position: Dict[str, Any], current_price: float) -> bool:
        """Return True if the position should be closed due to stop-loss."""
        entry_price = position.get("entry_price", 0)
        if not entry_price:
            logger.warning("Cannot calculate stop loss without entry price")
            return False

        stop_pct = self._stop_pct_for_position(position)
        direction = position.get("direction", "LONG")
        if direction == "LONG":
            loss_pct = (float(entry_price) - current_price) / float(entry_price)
        else:
            loss_pct = (current_price - float(entry_price)) / float(entry_price)

        if loss_pct >= stop_pct:
            logger.warning(
                f"Stop-loss triggered: {position.get('symbol', '?')} "
                f"({loss_pct * 100:.1f}% loss, stop={stop_pct * 100:.1f}%)"
            )
            return True
        return False
