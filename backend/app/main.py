from fastapi import FastAPI

from app.api import auth
from app.api import investigations
from app.api import documents

app = FastAPI(
    title="Agentic Real Estate Intelligence Platform",
    version="1.0.0"
)

app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

app.include_router(
    investigations.router,
    prefix="/api/v1/investigations",
    tags=["Investigations"]
)


@app.get("/")
def root():
    return {
        "status": "running",
        "application": "Real Estate Intelligence Platform"
    }

app.include_router(
    documents.router,
    prefix="/api/v1/investigations",
    tags=["Documents"]
)