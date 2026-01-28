#!/usr/bin/env python3
"""Logging configuration"""
import os
from dataclasses import dataclass

def _bool(val: str) -> bool:
    return val.lower() == "true"

@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Load logging config from environment"""
        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
