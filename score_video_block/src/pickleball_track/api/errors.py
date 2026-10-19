"""Map domain errors to HTTP responses with the FastAPI ``{"detail": ...}`` body."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..core.exceptions import (
    InvalidInputError,
    PayloadTooLargeError,
    PickleballError,
    ProcessingError,
    ServiceUnavailableError,
)

STATUS_CODES: dict[type[PickleballError], int] = {
    InvalidInputError: 422,
    PayloadTooLargeError: 413,
    ServiceUnavailableError: 503,
    ProcessingError: 502,
}


async def _handle_domain_error(request: Request, exc: PickleballError) -> JSONResponse:
    status_code = next(
        (code for error_type, code in STATUS_CODES.items() if isinstance(exc, error_type)),
        500,
    )
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(PickleballError, _handle_domain_error)
