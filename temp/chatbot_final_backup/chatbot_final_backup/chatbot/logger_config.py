import os
import logging
from config.constant import LOG_FILE

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

def setup_logger():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    # Create a logging format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger