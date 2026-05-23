"""Trade execution (writes to file for paper trading)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger


class TradeExecutor:
    """Execute trades (currently writes to file for paper trading)."""

    def __init__(self, enabled: bool = False, output_dir: str = "data/trades"):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trades:
            logger.info("No trades to execute")
            return {"executed": [], "failed": [], "timestamp": datetime.now().isoformat()}

        timestamp = datetime.now()
        results = {"executed": [], "failed": [], "timestamp": timestamp.isoformat()}

        for trade in trades:
            try:
                trade_with_metadata = self._add_trade_metadata(trade, timestamp)
                if self.enabled:
                    logger.warning("Real trade execution not implemented yet")
                    results["failed"].append(trade_with_metadata)
                else:
                    self._write_trade_to_file(trade_with_metadata, timestamp)
                    results["executed"].append(trade_with_metadata)
                    logger.info(
                        f"Trade recorded: {trade.get('symbol', '?')} "
                        f"({trade.get('direction', 'LONG')}) @ ${trade.get('entry_price', '?')}"
                    )
            except Exception as e:
                logger.error(f"Failed to execute trade: {e}")
                trade["error"] = str(e)
                results["failed"].append(trade)

        self._write_execution_summary(results, timestamp)
        logger.info(
            f"Execution complete: {len(results['executed'])} succeeded, "
            f"{len(results['failed'])} failed"
        )
        return results

    def _add_trade_metadata(self, trade: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        return {
            **trade,
            "execution_timestamp": timestamp.isoformat(),
            "execution_date": timestamp.strftime("%Y-%m-%d"),
            "execution_time": timestamp.strftime("%H:%M:%S"),
            "status": "open",
            "execution_method": "paper_trade",
        }

    def _write_trade_to_file(self, trade: Dict[str, Any], timestamp: datetime) -> None:
        date_str = timestamp.strftime("%Y%m%d")
        filename = self.output_dir / f"trades_{date_str}.jsonl"
        with open(filename, 'a') as f:
            f.write(json.dumps(trade, indent=2) + "\n\n")

    def _write_execution_summary(self, results: Dict[str, Any], timestamp: datetime) -> None:
        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M%S")
        filename = self.output_dir / f"summary_{date_str}_{time_str}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def save_rejected_trades(self, trades: List[Dict[str, Any]], reason: str) -> None:
        if not trades:
            return
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y%m%d")
        filename = self.output_dir / f"rejected_{date_str}.jsonl"
        with open(filename, 'a') as f:
            for trade in trades:
                record = {**trade, "rejected_at": timestamp.isoformat(), "rejection_reason": reason}
                f.write(json.dumps(record, indent=2) + "\n\n")
        logger.info(f"{len(trades)} rejected trades saved to {filename} (reason: {reason})")

    def close_position(self, position: Dict[str, Any]) -> bool:
        if not self.enabled:
            logger.info(f"Paper trade close: {position.get('symbol', '?')}")
            return True
        logger.warning("Live position close not yet implemented")
        return False

    def _load_trades_from_file(self, filename: Path) -> List[Dict[str, Any]]:
        trades = []
        decoder = json.JSONDecoder()
        content = filename.read_text()
        idx = 0
        while idx < len(content):
            content_slice = content[idx:].lstrip()
            if not content_slice:
                break
            try:
                obj, end = decoder.raw_decode(content_slice)
                trades.append(obj)
                idx += len(content) - len(content_slice) + end
            except json.JSONDecodeError:
                break
        return trades

    def load_todays_trades(self) -> List[Dict[str, Any]]:
        today_str = datetime.now().strftime("%Y%m%d")
        filename = self.output_dir / f"trades_{today_str}.jsonl"
        if not filename.exists():
            return []
        return self._load_trades_from_file(filename)
