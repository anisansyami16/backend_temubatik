from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

import numpy as np

from app.core.constants import CLASS_NAMES
from app.core.model_registry import get_model

from app.utils.file import save_upload_file

from app.utils.image import (
    read_image,
    resize_image,
    prepare_image_tensor
)

from app.utils.preprocessing import (
    preprocess_image
)

from app.services.gradcam_service import (
    generate_gradcam
)

from app.services.visualization_service import (
    build_overlay,
    build_visualizations
)

from app.services.explanation_service import (
    generate_explanation
)

router = APIRouter()


@router.post("/explain/{model_name}")
async def explain(
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

    model_data = get_model(model_name)

    if not model_data:
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )

    model = model_data["model"]

    image = read_image(
        str(file_path)
    )

    original_image = resize_image(image)

    image_tensor = prepare_image_tensor(original_image)

    image_tensor = preprocess_image(image_tensor, model_name)

    predictions = model.predict(
        image_tensor,
        verbose=0
)

    pred_index = int(
        np.argmax(
            predictions[0]
        )
    )

    predicted_class = CLASS_NAMES[pred_index]

    confidence = float(
        predictions[0][pred_index]
    )

    gradcam_result = generate_gradcam(
        model=model,
        processed_image=image_tensor,
        pred_index=pred_index,
        backbone_name=model_data["backbone"],
        last_conv_layer_name=model_data["last_conv_layer"]
    )

    heatmap = gradcam_result["raw_heatmap"]

    visualizations = build_visualizations(
        original_image,
        heatmap
    )

    explanation_result = (
        generate_explanation(
            heatmap=heatmap,
            predicted_class=predicted_class
        )
    )

    return {

        "predicted_class":predicted_class,

        "confidence":confidence,

        "predictions": predictions[0].tolist(),

        "focus_region":explanation_result["focus_region"],

        "focus_percentage":explanation_result["focus_percentage"],

        "peak_activation":explanation_result["peak_activation"],

        "attention_distribution":explanation_result["attention_distribution"],

        "active_regions":explanation_result["active_regions"],

        "region_scores":explanation_result["region_scores"],

        "explanation":explanation_result["explanation"],

        "overlay_image":visualizations["overlay_image"]
    }