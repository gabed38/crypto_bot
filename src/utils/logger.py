"""Logging setup for the crypto trading bot."""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    script_name: str = "crypto_bot",
):
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="{time:HH:mm:ss} | {level:<7} | {message}")

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path / f"{script_name}_{{time:YYYY-MM-DD}}.log"),
            level=log_level,
            rotation="1 day",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
        )

    return logger


def get_logger():
    return logger
