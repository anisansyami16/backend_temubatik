from tensorflow.keras.models import load_model

from app.core.constants import MODEL_CONFIGS

MODEL_REGISTRY = {}


def load_models():

    for model_key, config in MODEL_CONFIGS.items():

        print(f"Loading model: {model_key}")

        model = load_model(config["model_path"])

        MODEL_REGISTRY[model_key] = {
            "model": model,
            "display_name": config["display_name"],
            "backbone": config["backbone"],
            "last_conv_layer": config["last_conv_layer"],
        }

        print(f"Successfully loaded: {model_key}")


def get_model(model_name: str):

    return MODEL_REGISTRY.get(model_name)


def get_backbone_name(model_name: str):

    model_data = get_model(model_name)

    if not model_data:
        raise ValueError(
            f"Model {model_name} not found"
        )

    return model_data["backbone"]


def get_last_conv_layer(model_name: str):

    model_data = get_model(model_name)

    if not model_data:
        raise ValueError(
            f"Model {model_name} not found"
        )

    return model_data["last_conv_layer"]