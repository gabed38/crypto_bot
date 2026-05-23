"""Crypto price data from CoinGecko (free API, no key required)."""

import requests
from typing import Dict, Any, List, Optional
from loguru import logger


COINGECKO_BASE = "https://api.coingecko.com/api/v3"


class CryptoPriceClient:
    """Fetch crypto market data from CoinGecko free API."""

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_top_coins(self) -> List[Dict[str, Any]]:
        """Fetch top coins by market cap with price, volume, and change data.

        CoinGecko free API allows a maximum of 250 coins per page.  When
        ``top_n`` exceeds 250 the method automatically paginates, sleeping
        1.5 s between pages to respect the free-tier rate limit (~30 req/min).
        """
        import time as _time

        MAX_PER_PAGE = 250
        all_coins: List[Dict[str, Any]] = []
        remaining = self.top_n
        page = 1

        while remaining > 0:
            per_page = min(remaining, MAX_PER_PAGE)
            try:
                resp = self.session.get(
                    f"{COINGECKO_BASE}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": per_page,
                        "page": page,
                        "sparkline": "false",
                        "price_change_percentage": "1h,24h,7d",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                all_coins.extend(batch)
                remaining -= len(batch)
                page += 1
                if remaining > 0:
                    _time.sleep(1.5)  # rate-limit buffer between pages
            except Exception as e:
                logger.error(f"Failed to fetch coins page {page}: {e}")
                break

        logger.info(f"Fetched {len(all_coins)} coins from CoinGecko ({page - 1} page(s))")
        return all_coins

    def get_coin_price(self, coin_id: str) -> Optional[float]:
        """Get current USD price for a specific coin."""
        try:
            resp = self.session.get(
                f"{COINGECKO_BASE}/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get(coin_id, {}).get("usd")
        except Exception as e:
            logger.error(f"Failed to fetch price for {coin_id}: {e}")
            return None

    def get_coin_details(self, coin_id: str) -> Optional[Dict]:
        """Get detailed info for a single coin."""
        try:
            resp = self.session.get(
                f"{COINGECKO_BASE}/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "community_data": "false",
                    "developer_data": "false",
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch details for {coin_id}: {e}")
            return None

    def get_fear_greed_index(self) -> Dict[str, Any]:
        """Fetch the Crypto Fear & Greed Index from alternative.me."""
        try:
            resp = self.session.get(
                "https://api.alternative.me/fng/",
                params={"limit": 1},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [{}])[0]
            return {
                "value": int(data.get("value", 0)),
                "classification": data.get("value_classification", "Unknown"),
            }
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed Index: {e}")
            return {"value": 0, "classification": "Unknown"}

    def get_price_history(self, coin_id: str, days: int = 14) -> List[float]:
        """
        Fetch daily closing prices for volatility calculation.

        Uses CoinGecko's /coins/{id}/market_chart endpoint (free, no key).
        Returns a list of closing prices in USD, oldest first.
        Retries up to 3 times with exponential backoff on 429 rate-limit errors.
        """
        import time as _time
        for attempt in range(3):
            try:
                resp = self.session.get(
                    f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
                    params={"vs_currency": "usd", "days": days, "interval": "daily"},
                    timeout=15,
                )
                if resp.status_code == 429:
                    wait = 15 * (2 ** attempt)  # 15s, 30s, 60s
                    logger.warning(f"Rate-limited fetching {coin_id} history — waiting {wait}s (attempt {attempt + 1}/3)")
                    _time.sleep(wait)
                    continue
                resp.raise_for_status()
                prices = resp.json().get("prices", [])
                return [float(p[1]) for p in prices if len(p) >= 2]
            except Exception as e:
                if attempt < 2:
                    _time.sleep(5 * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch price history for {coin_id}: {e}")
        return []

    @staticmethod
    def _compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
        """
        Compute RSI using Wilder's smoothing from a list of closing prices.

        Args:
            closes: List of closing prices, oldest first (needs period + 1 minimum).
            period: Look-back period, default 14.

        Returns:
            RSI value (0–100) or None if insufficient data.
        """
        if len(closes) < period + 1:
            return None

        gains: List[float] = []
        losses: List[float] = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))

        # Seed with simple average over first `period` bars
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Wilder's smoothing for remaining bars
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 1)

    def get_intraday_rsi(
        self,
        coin_id: str,
        period: int = 14,
    ) -> Dict[str, Any]:
        """
        Fetch the last 24 hours of hourly OHLCV and compute RSI(14).

        Uses CoinGecko's /coins/{id}/market_chart endpoint with
        ``days=1&interval=hourly``, which returns ~24-25 data points on the
        free tier — just enough for RSI(14) with a few warm-up bars.

        Returns a dict with:
            rsi               — RSI(14) on the hourly chart (float), or None
            recent_4h_pct     — % change over the last 4 hourly closes (float)
            data_points       — number of hourly closes returned by the API (int)

        Returns an empty dict on any error so callers can treat missing RSI
        gracefully (skip rather than crash).
        """
        import time as _time
        try:
            resp = self.session.get(
                f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": 1, "interval": "hourly"},
                timeout=15,
            )
            if resp.status_code == 429:
                logger.warning(f"Rate-limited fetching intraday RSI for {coin_id} — waiting 15s")
                _time.sleep(15)
                resp = self.session.get(
                    f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
                    params={"vs_currency": "usd", "days": 1, "interval": "hourly"},
                    timeout=15,
                )
            resp.raise_for_status()
            prices = resp.json().get("prices", [])
            closes = [float(p[1]) for p in prices if len(p) >= 2]

            if not closes:
                return {}

            rsi = self._compute_rsi(closes, period)

            # % change over last 4 hourly closes (≈4h momentum)
            recent_4h: Optional[float] = None
            if len(closes) >= 4 and closes[-4] > 0:
                recent_4h = round((closes[-1] - closes[-4]) / closes[-4] * 100, 2)

            return {
                "rsi": rsi,
                "recent_4h_pct": recent_4h,
                "data_points": len(closes),
            }
        except Exception as e:
            logger.error(f"Failed to fetch intraday RSI for {coin_id}: {e}")
            return {}

    def get_trending_coins(self) -> List[Dict]:
        """Fetch trending coins from CoinGecko."""
        try:
            resp = self.session.get(f"{COINGECKO_BASE}/search/trending", timeout=10)
            resp.raise_for_status()
            items = resp.json().get("coins", [])
            return [item.get("item", {}) for item in items]
        except Exception as e:
            logger.error(f"Failed to fetch trending coins: {e}")
            return []

    def format_for_llm(self, coins: List[Dict], fear_greed: Dict, trending: List[Dict]) -> str:
        """Format crypto market data for LLM prompt injection.

        Accepts either raw coins or pre-screened coins (with screen_score fields).
        """
        lines = ["CRYPTO MARKET DATA:"]

        # Fear & Greed
        fg_val = fear_greed.get("value", 0)
        fg_class = fear_greed.get("classification", "Unknown")
        lines.append(f"\nFear & Greed Index: {fg_val}/100 ({fg_class})")

        # Trending
        if trending:
            lines.append("\nTRENDING COINS:")
            for t in trending[:7]:
                lines.append(f"  - {t.get('name', '?')} ({t.get('symbol', '?').upper()}) — Rank #{t.get('market_cap_rank', '?')}")

        # Coins table — adapt header based on whether screen data is present
        is_screened = any(c.get("screen_score") is not None for c in coins)

        has_vol = any(c.get("daily_vol_pct") is not None for c in coins)

        has_rsi = any(c.get("intraday_rsi") is not None for c in coins)

        if is_screened:
            lines.append(f"\nPRE-SCREENED CANDIDATES ({len(coins)} coins passed quantitative filter):")
            if has_vol and has_rsi:
                lines.append(
                    f"{'#':<4} {'Symbol':<8} {'Price':>12} {'24h%':>8} {'4h%':>7} "
                    f"{'MktCap':>12} {'Vol24h':>12} {'Score':>7} {'Vol/day':>8} {'Stop':>7} {'RSI':>5} {'Signals'}"
                )
                lines.append("-" * 125)
            elif has_vol:
                lines.append(
                    f"{'#':<4} {'Symbol':<8} {'Price':>12} {'24h%':>8} {'7d%':>8} "
                    f"{'MktCap':>12} {'Vol24h':>12} {'Score':>7} {'Vol/day':>8} {'StopMult':>9} {'Signals'}"
                )
                lines.append("-" * 115)
            else:
                lines.append(
                    f"{'#':<4} {'Symbol':<8} {'Price':>12} {'24h%':>8} {'7d%':>8} "
                    f"{'MktCap':>12} {'Vol24h':>12} {'Score':>7} {'Signals'}"
                )
                lines.append("-" * 100)
        else:
            lines.append(f"\nTOP {len(coins)} COINS BY MARKET CAP:")
            lines.append(f"{'#':<4} {'Symbol':<8} {'Price':>12} {'24h%':>8} {'7d%':>8} {'MktCap':>14} {'Vol24h':>14}")
            lines.append("-" * 72)

        for coin in coins:
            rank = coin.get("market_cap_rank", "?")
            symbol = (coin.get("symbol") or "?").upper()
            price = coin.get("current_price") or 0
            change_24h = coin.get("price_change_percentage_24h") or 0
            change_7d = coin.get("price_change_percentage_7d_in_currency") or 0
            mcap = coin.get("market_cap") or 0
            vol = coin.get("total_volume") or 0

            if price >= 1:
                price_str = f"${price:,.2f}"
            elif price >= 0.01:
                price_str = f"${price:.4f}"
            else:
                price_str = f"${price:.6f}"

            mcap_str = f"${mcap / 1e9:.1f}B" if mcap >= 1e9 else f"${mcap / 1e6:.0f}M"
            vol_str = f"${vol / 1e9:.1f}B" if vol >= 1e9 else f"${vol / 1e6:.0f}M"

            if is_screened:
                score = coin.get("screen_score", 0)
                reasons = ", ".join(coin.get("screen_reasons", [])) or "baseline"
                if has_vol and has_rsi:
                    dv = coin.get("daily_vol_pct")
                    sm = coin.get("stop_multiple")
                    rsi_val = coin.get("intraday_rsi")
                    r4h = coin.get("recent_4h_pct")
                    dv_str = f"{dv:.1f}%" if dv is not None else "N/A"
                    sm_str = f"{sm:.1f}x" if sm is not None else "N/A"
                    rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else "N/A"
                    r4h_str = f"{r4h:+.1f}%" if r4h is not None else "N/A"
                    lines.append(
                        f"{rank:<4} {symbol:<8} {price_str:>12} {change_24h:>+7.1f}% {r4h_str:>6} "
                        f"{mcap_str:>12} {vol_str:>12} {score:>+6.1f}  {dv_str:>7} {sm_str:>6} {rsi_str:>5}  {reasons}"
                    )
                elif has_vol:
                    dv = coin.get("daily_vol_pct")
                    sm = coin.get("stop_multiple")
                    dv_str = f"{dv:.1f}%" if dv is not None else "N/A"
                    sm_str = f"{sm:.1f}x" if sm is not None else "N/A"
                    lines.append(
                        f"{rank:<4} {symbol:<8} {price_str:>12} {change_24h:>+7.1f}% {change_7d:>+7.1f}% "
                        f"{mcap_str:>12} {vol_str:>12} {score:>+6.1f}  {dv_str:>7} {sm_str:>8}  {reasons}"
                    )
                else:
                    lines.append(
                        f"{rank:<4} {symbol:<8} {price_str:>12} {change_24h:>+7.1f}% {change_7d:>+7.1f}% "
                        f"{mcap_str:>12} {vol_str:>12} {score:>+6.1f}  {reasons}"
                    )
            else:
                lines.append(
                    f"{rank:<4} {symbol:<8} {price_str:>12} {change_24h:>+7.1f}% {change_7d:>+7.1f}% {mcap_str:>14} {vol_str:>14}"
                )

        return "\n".join(lines)
