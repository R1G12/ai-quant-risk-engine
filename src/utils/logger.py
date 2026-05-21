'''logger.py – structured JSON logger for the quant risk engine'''

import json
import logging
import sys
from pathlib import Path
from typing import Final

# Resolve the repository root (two levels up from this file)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# Default log directory – can be overridden by the LOG_DIR env var
LOG_DIR: Final[Path] = Path(
    sys.getenv("LOG_DIR", str(PROJECT_ROOT / "logs"))
).expanduser().resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Basic JSON formatter – each log record is a single JSON object per line
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def get_logger(name: str = "quant_engine") -> logging.Logger:
    """Return a configured logger.

    The logger writes JSON lines to ``<repo>/logs/<name>.log`` and also
    streams to ``stdout``. Handlers are added only once per process.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger already configured – avoid duplicate handlers
        return logger

    logger.setLevel(logging.INFO)

    # File handler (JSON lines)
    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    # Console handler – human‑readable
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    return logger
