"""Crypto news aggregation from RSS feeds."""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, List
import requests
from loguru import logger


RSS_FEEDS = {
    # Tier 1 — high quality, broad coverage
    "coindesk":        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph":   "https://cointelegraph.com/rss",
    "decrypt":         "https://decrypt.co/feed",
    "theblock":        "https://www.theblock.co/rss.xml",
    # Tier 2 — good altcoin and market coverage
    "beincrypto":      "https://beincrypto.com/feed/",
    "bitcoinist":      "https://bitcoinist.com/feed/",
    "newsbtc":         "https://www.newsbtc.com/feed/",
    "ambcrypto":       "https://ambcrypto.com/feed/",
    # Tier 3 — additional breadth
    "bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
    "cryptopotato":    "https://cryptopotato.com/feed/",
}


class CryptoNewsClient:
    """Aggregate crypto news from RSS feeds."""

    def __init__(self, max_articles: int = 30):
        self.max_articles = max_articles
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CryptoBot/1.0",
            "Accept": "application/xml, application/rss+xml, text/xml",
        })

    def get_rss_news(self) -> List[Dict[str, Any]]:
        """Fetch news from crypto RSS feeds."""
        articles = []
        for source, url in RSS_FEEDS.items():
            try:
                resp = self.session.get(url, timeout=10)
                resp.raise_for_status()
                root = ET.fromstring(resp.content)

                # Handle both RSS and Atom formats
                items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
                for item in items[:6]:
                    title = self._get_text(item, "title") or self._get_text(
                        item, "{http://www.w3.org/2005/Atom}title"
                    )
                    description = self._get_text(item, "description") or self._get_text(
                        item, "{http://www.w3.org/2005/Atom}summary"
                    )
                    pub_date = self._get_text(item, "pubDate") or self._get_text(
                        item, "{http://www.w3.org/2005/Atom}published"
                    )

                    if title:
                        articles.append({
                            "source": source,
                            "title": title.strip(),
                            "description": (description or "")[:300].strip(),
                            "published": pub_date or "",
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch RSS from {source}: {e}")

        logger.info(f"Fetched {len(articles)} articles from RSS feeds")
        return articles[:self.max_articles]

    def get_all_news(self) -> List[Dict[str, Any]]:
        """Fetch and deduplicate news from all RSS sources."""
        all_news = self.get_rss_news()
        seen_titles = set()
        unique = []
        for article in all_news:
            key = article["title"][:60].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(article)
        return unique[:self.max_articles]

    def filter_by_coins(
        self,
        articles: List[Dict[str, Any]],
        coin_symbols: List[str],
        coin_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Filter articles to only those mentioning shortlisted coins.

        Matches against coin symbols (BTC, ETH) and names (Bitcoin, Ethereum)
        in the article title and description. Also keeps up to 5 general market
        articles that don't mention specific coins but affect the whole market.
        """
        if not coin_symbols and not coin_names:
            return articles

        # Build keyword set (lowercase)
        keywords = set()
        for s in coin_symbols:
            if len(s) >= 3:  # skip very short symbols to avoid false matches
                keywords.add(s.lower())
        for n in coin_names:
            keywords.add(n.lower())

        # General market keywords — keep articles about overall crypto market
        market_keywords = {
            "crypto market", "bitcoin etf", "crypto etf", "federal reserve",
            "interest rate", "sec ", "regulation", "crypto regulation",
            "bull market", "bear market", "crypto crash", "crypto rally",
            "market cap", "defi", "stablecoin",
        }

        relevant = []
        general = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()

            if any(kw in text for kw in keywords):
                relevant.append(article)
                continue

            if any(kw in text for kw in market_keywords):
                general.append(article)

        result = relevant + general[:5]
        logger.info(
            f"News filter: {len(relevant)} coin-specific + "
            f"{min(len(general), 5)} general market = {len(result)} articles "
            f"(from {len(articles)} total)"
        )
        return result

    def format_for_llm(self, articles: List[Dict]) -> str:
        """Format news articles for LLM prompt injection."""
        if not articles:
            return "CRYPTO NEWS: No recent news available."

        lines = [f"RECENT CRYPTO NEWS ({len(articles)} articles):"]
        for i, article in enumerate(articles, 1):
            source = article.get("source", "?")
            title = article.get("title", "")
            desc = article.get("description", "")

            line = f"\n{i}. [{source.upper()}] {title}"
            if desc:
                line += f"\n   {desc[:200]}"
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _get_text(element, tag: str) -> str:
        child = element.find(tag)
        return child.text if child is not None and child.text else ""
