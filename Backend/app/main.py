"""FastAPI application entry point."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, personalization, products
from app.services.embeddings import get_embedding_model
from app.services.database import initialize_demo_user_storage
from app.services.personalization import DEMO_PROFILES
from app.services.vector_search import load_current_index, validate_catalog_against_index

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


async def _validate_catalog_periodically() -> None:
    """Check index freshness off the request path every five minutes."""
    while True:
        await asyncio.sleep(300)
        try:
            current = await asyncio.to_thread(validate_catalog_against_index)
            if not current:
                logger.error("FAISS index is stale relative to MongoDB; semantic search is paused until reindexing.")
        except Exception as exc:
            logger.warning("Could not validate catalog/index freshness (%s).", type(exc).__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load expensive local search resources once, before serving requests."""
    started = time.perf_counter()
    stage = time.perf_counter()
    await asyncio.to_thread(get_embedding_model)
    model_load = time.perf_counter() - stage
    stage = time.perf_counter()
    await asyncio.to_thread(load_current_index)
    index_load = time.perf_counter() - stage
    stage = time.perf_counter()
    current = await asyncio.to_thread(validate_catalog_against_index)
    catalog_validation = time.perf_counter() - stage
    stage = time.perf_counter()
    await asyncio.to_thread(initialize_demo_user_storage, DEMO_PROFILES)
    demo_user_setup = time.perf_counter() - stage
    if not current:
        logger.error("FAISS index is stale relative to MongoDB; semantic search will return a service error.")
    logger.info(
        "Search startup timings: model_load_s=%.4f index_load_s=%.4f catalog_validation_s=%.4f demo_user_setup_s=%.4f total_s=%.4f",
        model_load,
        index_load,
        catalog_validation,
        demo_user_setup,
        time.perf_counter() - started,
    )
    validator = asyncio.create_task(_validate_catalog_periodically())
    try:
        yield
    finally:
        validator.cancel()
        try:
            await validator
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title="Personalized E-Commerce Search API",
    description="Initial API for health checks and reading the product catalog.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Vite uses 5173 by default and may fall back to 5174 when that port is busy.
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    """Welcome route for the API."""
    return {"message": "Welcome to the Personalized E-Commerce Search API."}


app.include_router(health.router)
app.include_router(products.router)
app.include_router(personalization.router)


@app.middleware("http")
async def log_search_request_time(request, call_next):
    """Measure complete search request time, including response serialization."""
    if request.url.path != "/api/products/search":
        return await call_next(request)
    started = time.perf_counter()
    response = await call_next(request)
    logger.info("Search HTTP request completed in %.4fs (status=%s).", time.perf_counter() - started, response.status_code)
    return response
