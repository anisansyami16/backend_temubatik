import numpy as np

from app.core.constants import CLASS_NAMES
from app.core.model_registry import get_model

from app.utils.image import (
    read_image,
    resize_image,
    prepare_image_tensor
)

from app.utils.preprocessing import (
    preprocess_image
)


def prepare_input(
    image_path: str,
    model_name: str
):

    image = read_image(image_path)

    original_image = resize_image(image)

    image_tensor = prepare_image_tensor(
        original_image
    )

    image_tensor = preprocess_image(
        image_tensor,
        model_name
    )

    return original_image, image_tensor


def predict_tensor(
    image_tensor,
    model_name: str
):

    model_data = get_model(model_name)

    if not model_data:
        raise ValueError(
            f"Model '{model_name}' not found."
        )

    model = model_data["model"]

    predictions = model.predict(
        image_tensor,
        verbose=0
    )

    pred_index = int(
        np.argmax(predictions[0])
    )

    confidence = float(
        predictions[0][pred_index]
    )

    predicted_class = CLASS_NAMES[
        pred_index
    ]

    return {

        "model": model,

        "model_data": model_data,

        "predictions": predictions,

        "pred_index": pred_index,

        "predicted_class": predicted_class,

        "confidence": confidence

    }


def predict_image(
    image_path: str,
    model_name: str
):

    _, image_tensor = prepare_input(
        image_path,
        model_name
    )

    result = predict_tensor(
        image_tensor,
        model_name
    )

    return {

        "predicted_class":
            result["predicted_class"],

        "confidence":
            result["confidence"],

        "predictions":
            result["predictions"][0].tolist()

    }