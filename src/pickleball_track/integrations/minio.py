import os
from urllib.parse import urlsplit

from fastapi import HTTPException


def create_minio_client(endpoint: str):
    try:
        from minio import Minio
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="MinIO SDK is missing; install the project dependencies",
        ) from exc

    parsed = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    if not parsed.hostname or parsed.path not in ("", "/"):
        raise HTTPException(status_code=422, detail="Invalid minio_endpoint")
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=422,
            detail="minio_endpoint must use HTTP or HTTPS",
        )

    secure = parsed.scheme == "https" or (
        not parsed.scheme and os.getenv("MINIO_SECURE", "false").lower() == "true"
    )
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    if not access_key or not secret_key:
        raise HTTPException(
            status_code=503,
            detail="MinIO credentials are not configured",
        )

    return Minio(
        parsed.netloc or parsed.path,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )