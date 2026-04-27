import logging
import os
from datetime import datetime


def setup_logger(class_name: str, run_id:str, log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{run_id}.log")

    logger = logging.getLogger(class_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger