
import logging
import logging.config
from pathlib import Path

# ANSI color codes for log levels
LOG_COLORS = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Green
    'WARNING': '\033[33m',    # Yellow
    'ERROR': '\033[31m',      # Red
    'CRITICAL': '\033[41m',   # Red background
}
RESET_COLOR = '\033[0m'

class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{RESET_COLOR}"
        return super().format(record)

# Resolve logging.conf relative to this file
config_path = Path(__file__).parent.parent / "logging.conf"
logging.config.fileConfig(config_path)
log = logging.getLogger('dev')

# Patch console handler to use ColorFormatter
for handler in log.handlers:
    if isinstance(handler, logging.StreamHandler):
        fmt = handler.formatter._fmt if hasattr(handler.formatter, '_fmt') else None
        datefmt = handler.formatter.datefmt if hasattr(handler.formatter, 'datefmt') else None
        handler.setFormatter(ColorFormatter(fmt, datefmt))

def get_logger():
    """Returns the log class for using the same logger in other files"""
    return log
