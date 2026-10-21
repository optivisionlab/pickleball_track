"""Domain errors raised by services; the API layer maps them to HTTP statuses."""


class PickleballError(Exception):
    """Base class for errors with a client-safe message."""


class InvalidInputError(PickleballError):
    """The request is well-formed but its values cannot be processed (422)."""


class PayloadTooLargeError(PickleballError):
    """The referenced video exceeds the configured size limit (413)."""


class ServiceUnavailableError(PickleballError):
    """A required model or dependency is not configured or not loaded (503)."""


class ServiceBusyError(PickleballError):
    """Too many requests in progress; the client should retry after ``retry_after`` (429)."""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class ProcessingError(PickleballError):
    """Downloading the video or running inference failed (502)."""


class ModelLoadError(PickleballError):
    """Model artifacts are missing or incompatible with the architecture."""
