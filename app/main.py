from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.router import api_router
from app.core.model_registry import load_models


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting Temu Batik API...")
    load_models()
    print("All models loaded successfully")

    yield

    print("Shutting down application...")


app = FastAPI(
    title="Temu Batik API",
    description="Backend API for Batik Classification and Explainable AI",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router)