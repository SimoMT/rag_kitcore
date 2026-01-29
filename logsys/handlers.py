import logging
import os
from .formatter import get_formatter

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_console_handler() -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(get_formatter())
    return handler

def get_file_handler() -> logging.Handler:
    handler = logging.FileHandler(f"{LOG_DIR}/app.log")
    handler.setFormatter(get_formatter())
    return handler