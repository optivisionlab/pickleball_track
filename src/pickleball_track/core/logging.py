"""Minimal JSON logging setup for API and inference events."""

from datetime import datetime, timezone
import json
import logging
import os


class JsonFormatter(logging.Formatter):
    _extra_fields = (
        "device",
        "duration_seconds",
        "fps",
        "frames",
        "hit_count",
        "job_id",
        "miss_count",
        "model",
        "score",
        "window_count",
        "video_duration_seconds",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self._extra_fields:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        handlers=[handler],
        force=True,
    )