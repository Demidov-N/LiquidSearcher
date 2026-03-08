"""Logging utilities for the Liquidity Risk Management project."""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = logging.INFO


def get_logger(
    name: str,
    level: int = DEFAULT_LOG_LEVEL,
    log_format: str | None = None,
) -> logging.Logger:
    """Get or create a configured logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: INFO).
        log_format: Custom format string (default: LOG_FORMAT).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt=log_format or LOG_FORMAT,
            datefmt=LOG_DATE_FORMAT,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def set_log_level(logger: logging.Logger, level: int) -> None:
    """Set log level for a logger and all its handlers.

    Args:
        logger: Logger to modify.
        level: New logging level.
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def add_file_handler(
    logger: logging.Logger,
    filepath: str,
    level: int = DEFAULT_LOG_LEVEL,
    log_format: str | None = None,
) -> logging.FileHandler:
    """Add a file handler to an existing logger.

    Args:
        logger: Logger to add handler to.
        filepath: Path to log file.
        level: Logging level for file handler.
        log_format: Custom format string.

    Returns:
        Created file handler.
    """
    handler = logging.FileHandler(filepath, mode="a")
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt=log_format or LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return handler
