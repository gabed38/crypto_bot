"""LLM-powered crypto market analysis using Claude."""

import json
import time
from typing import Dict, Any, List
from anthropic import Anthropic, APIStatusError
from loguru import logger


class LLMAnalyzer:
    """Use Claude to analyze crypto markets and make trading decisions."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def analyze(
        self,
        prompt: str,
        system: str = None,
        model: str = None,
    ) -> Dict[str, Any]:
        """Send a prompt to Claude and return parsed JSON response.

        If `system` is provided, it is sent as a cached system prompt
        (cache_control: ephemeral) so stable strategy text doesn't re-bill
        on every run. If `model` is provided, it overrides the instance default.

        Retries up to 3 times on 529 overloaded errors with exponential backoff.
        """
        use_model = model or self.model
        request_kwargs: Dict[str, Any] = {
            "model": use_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request_kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        for attempt in range(3):
            try:
                logger.info(
                    f"Sending analysis request to Claude "
                    f"(model={use_model}, cached_system={bool(system)}, attempt {attempt + 1})"
                )
                response = self.client.messages.create(**request_kwargs)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
                    if cache_read or cache_create:
                        logger.info(
                            f"Cache tokens — read: {cache_read}, created: {cache_create}"
                        )
                analysis = self._parse_json_response(response.content[0].text)
                logger.info(f"LLM recommended {len(analysis.get('trades', []))} trades")
                return analysis
            except APIStatusError as e:
                if e.status_code == 529 and attempt < 2:
                    wait = 20 * (attempt + 1)  # 20s, 40s
                    logger.warning(f"Claude overloaded (529) — retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                logger.error(f"LLM analysis failed: {e}")
                break
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                break

        return {
            "trades": [],
            "market_summary": "Analysis failed",
            "overall_sentiment": "neutral",
        }

    def analyze_performance(
        self,
        trades: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
        prompt_template: str,
    ) -> Dict[str, Any]:
        """Analyze trading performance and generate insights."""
        try:
            trades_text = json.dumps(trades, indent=2)
            positions_text = json.dumps(positions, indent=2)
            prompt = prompt_template.format(trades=trades_text, positions=positions_text)

            logger.info("Sending performance analysis request to Claude")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            analysis = self._parse_json_response(response.content[0].text)
            logger.info("Performance analysis complete")
            return analysis
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {
                "summary": {},
                "wins": [],
                "losses": [],
                "lessons": [],
                "recommendations": [],
                "error": str(e),
            }

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text}")
            return {
                "trades": [],
                "error": "Failed to parse LLM response",
                "raw_response": response_text,
            }
