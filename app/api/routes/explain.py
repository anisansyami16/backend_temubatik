from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.utils.file import save_upload_file

from app.services.explain_service import (
    explain_image
)

router = APIRouter()


@router.post("/explain/{model_name}")
async def explain(

    model_name: str,

    file: UploadFile = File(...)
):

    if model_name not in [
        "mobilenetv2",
        "resnet50"
    ]:

        raise HTTPException(
            status_code=400,
            detail="Invalid model name"
        )

    file_path = await save_upload_file(file)

    try:

        return explain_image(

            image_path=str(file_path),

            model_name=model_name
        )

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e)
        )