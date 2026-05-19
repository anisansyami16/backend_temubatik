import cv2
import numpy as np

from PIL import Image
from app.core.constants import IMAGE_SIZE


def read_image(image_path: str):

    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)

    return image


def resize_image(image: np.ndarray):

    image = cv2.resize(image, IMAGE_SIZE)

    return image


def prepare_image_tensor(image: np.ndarray):

    image = image.astype("float32")
    image = np.expand_dims(image, axis=0)

    return image