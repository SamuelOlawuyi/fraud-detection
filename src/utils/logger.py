"""Logging configuration for the fraud detection system."""

import sys
from loguru import logger
from pathlib import Path

from src.config import LOGS_DIR, settings


def setup_logger():
    """Configure and setup logger."""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        LOGS_DIR / "fraud_detection_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    return logger


def get_logger():
    """Get configured logger instance."""
    return setup_logger()