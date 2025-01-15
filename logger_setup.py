import os
import re
import logging
from logging.handlers import TimedRotatingFileHandler
import logging.config

def setup_logger(name: str, log_level: int = logging.INFO, log_file_prefix: str = "app", config: dict = None) -> logging.Logger:
    """
    Sets up a logger with separate file and console handlers using dictionary configuration.

    Args:
        name (str): Name of the logger.
        log_level (int): The logging level, defaults to logging.INFO.
        log_file_prefix (str): File name prefix for log files.
        config (dict): Configuration dictionary for logging.

    Returns:
        logging.Logger: The configured logger.
    """
    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'colored': {
                '()': 'logger_setup.ColorFormatter',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'plain': {
                '()': 'logger_setup.StripAnsiFormatter',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'colored',
                'level': log_level,
            },
            'file': {
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': f"{log_file_prefix}.log",
                'when': 'midnight',
                'interval': 1,
                'backupCount': 3,
                'formatter': 'plain'
            }
        },
        'loggers': {
            name: {
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': True
            }
        }
    }

    if config:
        default_config.update(config)

    logging.config.dictConfig(default_config)
    logger = logging.getLogger(name)
    return logger

class StripAnsiFormatter(logging.Formatter):
    """
    Custom formatter to remove ANSI escape sequences from log messages for file output.
    """
    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def format(self, record):
        original_message = super().format(record)
        return self.ANSI_ESCAPE.sub('', original_message)

class ColorFormatter(logging.Formatter):
    """
    Adds color codes to log messages for console output, making it more readable.
    """

    COLORS = {
        'DEBUG': '\033[94m',   # Blue
        'INFO': '\033[92m',    # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91;1m'  # Bright Red (with bold)
    }
    RESET = '\033[0m'

    def format(self, record):
        log_fmt = self.COLORS.get(record.levelname, "") + \
                  "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + \
                  self.RESET
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def main():
    # Set up logging
    log_level = logging.INFO
    log_file_prefix = "trading_analysis"
    config = {}  # You can define your logging configuration here
    logger = setup_logger("trading_analysis_log", log_level, log_file_prefix, config)

    # Your application code here
    # For example, you can log some test messages
    for level in logging.getLevelNames():
        logger.log(getattr(logging, level), f"This is a {level} message.")

if __name__ == "__main__":
    main()