import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


def preprocess_image(
    image: np.ndarray,
    model_name: str
):

    if model_name == "mobilenetv2":
        return mobilenet_preprocess(image)

    elif model_name == "resnet50":
        return resnet_preprocess(image)

    raise ValueError(
        f"Unsupported model: {model_name}"
    )