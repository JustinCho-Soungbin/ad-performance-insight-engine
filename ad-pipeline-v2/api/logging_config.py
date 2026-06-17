"""
logging_config.py
─────────────────────────────────────────
Structured (JSON) logging — a prerequisite for any future
observability work (Prometheus/Grafana). Every log line is a
machine-parseable JSON object instead of a free-text string.
"""

import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Include any extra fields passed via logger.info(msg, extra={...})
        reserved = vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
        for key, value in vars(record).items():
            if key not in reserved and key not in payload:
                payload[key] = value
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
