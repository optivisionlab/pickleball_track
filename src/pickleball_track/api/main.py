from fastapi import FastAPI

from ..core.logging import configure_logging
from .routes import router


configure_logging()

app = FastAPI(title="Pickleball AI Pipeline", version="1.0.0")
app.include_router(router)