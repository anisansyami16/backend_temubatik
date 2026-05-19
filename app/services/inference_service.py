import numpy as np

from app.core.constants import CLASS_NAMES
from app.core.model_registry import get_model

from app.utils.image import (
    read_image,
    resize_image,
    prepare_image_tensor
)

from app.utils.preprocessing import preprocess_image


def predict_image(
    image_path: str,
    model_name: str
):

    model_data = get_model(model_name)

    if not model_data:
        raise ValueError(
            f"Model {model_name} not found"
        )

    model = model_data["model"]

    image = read_image(image_path)
    image = resize_image(image)
    image = prepare_image_tensor(image)
    image = preprocess_image(image, model_name)

    predictions = model.predict(image)
    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index])
    predicted_class = CLASS_NAMES[predicted_index]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "predictions": predictions[0].tolist()
    }