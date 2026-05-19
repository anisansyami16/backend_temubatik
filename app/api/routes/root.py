from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():

    return {
        "message": "Temu Batik Backend is Running"
    }