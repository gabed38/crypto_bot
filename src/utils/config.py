"""Configuration management for the crypto trading bot."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from string import Template


class Config:
    """Load and manage configuration from YAML files with environment variable substitution."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            content = f.read()
        content = Template(content).safe_substitute(os.environ)
        return yaml.safe_load(content)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation. Example: config.get('llm.model')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        return key in self.config
