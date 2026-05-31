import os
import logging
from typing import Optional


def setup_logger(
    name: str = "fraud_pipeline",
    log_file: str = "logs/pipeline.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Sets up a centralized thread-safe logger with console and file handlers.

    Args:
        name: Name of the logger.
        log_file: Path where the log file will be saved.
        level: Logger logging level.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if the logger has already been configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format pattern
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to stdout if file writing is restricted
        print(f"Warning: Could not create file handler for logging due to: {e}")

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    return logger


# Primary instance for import
logger = setup_logger()
