import logging
import sys

def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = False,
    file_path: str = "app.log"
) -> logging.Logger:
    """
    Create or return a configured logger.

    - Prevents duplicate handlers
    - Supports console + optional file logging
    - Allows dynamic log level
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
