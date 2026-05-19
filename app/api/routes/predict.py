from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.schemas.prediction import PredictionResponse
from app.services.inference_service import predict_image
from app.utils.file import save_upload_file

router = APIRouter()


@router.post(
    "/predict/{model_name}",
    response_model=PredictionResponse
)
async def predict(
    model_name: str,
    file: UploadFile = File(...)
):

    file_path = await save_upload_file(file)

    if model_name not in [
      "mobilenetv2",
      "resnet50"
    ]:
      raise HTTPException(
          status_code=400,
          detail="Invalid model name"
      )

    result = predict_image(
        str(file_path),
        model_name
    )

    return result