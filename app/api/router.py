from fastapi import APIRouter
from app.api.routes.root import router as root_router
from app.api.routes.health import router as health_router
from app.api.routes.debug import router as debug_router
from app.api.routes.predict import router as predict_router
from app.api.routes.explain import router as explain_router

api_router = APIRouter()


api_router.include_router(
    root_router,
    tags=["Root"]
)

api_router.include_router(
    health_router,
    tags=["Health"]
)

api_router.include_router(
    debug_router,
    tags=["Debug"]
)

api_router.include_router(
    predict_router,
    tags=["Prediction"]
)

api_router.include_router(
    explain_router,
    tags=["Explain"]
)