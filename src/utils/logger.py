"""Logging utility module."""

import logging
from rich.logging import RichHandler
from rich.console import Console

console = Console()


def setup_logger(name: str = "mmwave", level: int = logging.INFO) -> logging.Logger:
    """Set up logger.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Set up rich handler
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        enable_link_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    # Set up formatter
    formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(formatter)
    rich_handler.setLevel(level)
    logger.addHandler(rich_handler)

    return logger


# Create default logger
logger = setup_logger()
