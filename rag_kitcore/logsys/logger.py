import logging
from rag_kitcore.logsys.handlers import get_console_handler, get_file_handler

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler())
        logger.propagate = False

    return logger