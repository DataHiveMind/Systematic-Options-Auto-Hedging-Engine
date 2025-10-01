"""
=======================================================================
Title: logging_utils.py
Author: Kenneth LeGare
Date: 2023-10-05

Description:
    This module sets up logging for the hedging engine application.
    It configures log formatting, log levels, and log file handling.
    Logs are saved to a file named 'hedging_engine.log'.
Dependencies:
    - logging
    - os
    - sys
==========================================================================
"""
import logging
import os
import sys
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "hedging_engine.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_start():
    logger.info("Hedging Engine started.")

def log_stop():
    logger.info("Hedging Engine stopped.")

def log_error(error_msg):
    logger.error(f"Error: {error_msg}")

def log_info(info_msg):
    logger.info(info_msg)

def log_debug(debug_msg):
    logger.debug(debug_msg)

def log_warning(warning_msg):
    logger.warning(warning_msg)

def log_critical(critical_msg):
    logger.critical(critical_msg)

# Example usage
if __name__ == "__main__":
    log_start()
    try:
        # Simulate some operations
        log_info("Performing some operations...")
        # Simulate a warning
        log_warning("This is a warning message.")
        # Simulate an error
        raise ValueError("This is a simulated error.")
    except Exception as e:
        log_error(str(e))
    finally:
        log_stop()