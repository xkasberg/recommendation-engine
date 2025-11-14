from functools import lru_cache

from fastapi import FastAPI
from contextlib import asynccontextmanager

from services.recommender import RecommenderService
from routers.recommender import router as recommender_router

from utils.logger import Logger

_logger = Logger(name="plush-recommendation_service")

@lru_cache(maxsize=None)
def get_recommender_service():
    """Instantiate (and cache) the singleton recommender service."""
    return RecommenderService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook that hydrates/tears down the recommender once."""
    app.state.recommender_service = get_recommender_service()
    _logger.info("Starting Application...")
    yield
    _logger.info("Shutting Down Application...")


app = FastAPI(title="Plush For You API", lifespan=lifespan)

app.include_router(recommender_router)

@app.get("/healthz")
async def health():
    """Simple readiness endpoint."""
    return {"ok": True, "greeting": "Eternal fire, that inward burns"}

