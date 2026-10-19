from dataclasses import replace

import pytest

from pickleball_track.core.exceptions import InvalidInputError, ServiceUnavailableError
from pickleball_track.integrations.minio import create_minio_client


@pytest.mark.parametrize(
    ("endpoint", "secure"),
    [("http://minio:9000", False), ("https://minio:9000", True), ("minio:9000", False)],
)
def test_client_uses_endpoint_scheme(test_settings, endpoint, secure):
    client = create_minio_client(endpoint, test_settings)

    assert client._base_url.is_https is secure


def test_endpoint_without_scheme_follows_minio_secure(test_settings):
    client = create_minio_client("minio:9000", replace(test_settings, minio_secure=True))

    assert client._base_url.is_https is True


@pytest.mark.parametrize("endpoint", ["ftp://minio:9000", "http://minio:9000/path", ""])
def test_invalid_endpoint_is_rejected(test_settings, endpoint):
    with pytest.raises(InvalidInputError):
        create_minio_client(endpoint, test_settings)


def test_missing_credentials_is_service_unavailable(test_settings):
    with pytest.raises(ServiceUnavailableError, match="credentials"):
        create_minio_client("minio:9000", replace(test_settings, minio_secret_key=None))
