from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from ..core.config import Settings
from ..core.exceptions import InvalidInputError, ServiceUnavailableError


def create_minio_client(endpoint: str, settings: Settings) -> Any:
    try:
        from minio import Minio
    except ImportError as exc:
        raise ServiceUnavailableError(
            "MinIO SDK is missing; install the project dependencies"
        ) from exc

    parsed = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    if not parsed.hostname or parsed.path not in ("", "/"):
        raise InvalidInputError("Invalid minio_endpoint")
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        raise InvalidInputError("minio_endpoint must use HTTP or HTTPS")

    secure = parsed.scheme == "https" or (not parsed.scheme and settings.minio_secure)
    if not settings.minio_access_key or not settings.minio_secret_key:
        raise ServiceUnavailableError("MinIO credentials are not configured")

    return Minio(
        parsed.netloc or parsed.path,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=secure,
    )
