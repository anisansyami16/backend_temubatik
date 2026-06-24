import uuid
import cv2
import numpy as np

from app.core.config import HEATMAP_DIR


def resize_heatmap(
    heatmap: np.ndarray,
    target_shape: tuple
):

    height, width = target_shape[:2]

    return cv2.resize(
        heatmap,
        (width, height)
    )


def apply_colormap(
    heatmap: np.ndarray
):

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_INFERNO
    )

    return heatmap_color


def build_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
):
    
    heatmap = resize_heatmap(heatmap, original_image.shape)

    threshold = 0.15

    mask = heatmap >= threshold

    filtered_heatmap = np.zeros_like(heatmap)

    filtered_heatmap[mask] = heatmap[mask]

    heatmap_color = apply_colormap(filtered_heatmap)

    original_bgr = cv2.cvtColor(
        original_image,
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        original_bgr,
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay


def save_overlay(
    overlay: np.ndarray
):
    
    filename = (
        f"overlay_{uuid.uuid4()}.jpg"
    )

    output_path = HEATMAP_DIR / filename

    cv2.imwrite(
        str(output_path),
        overlay
    )

    return output_path


def save_heatmap(
    heatmap: np.ndarray
):
    
    filename = (
        f"heatmap_{uuid.uuid4()}.jpg"
    )

    output_path = HEATMAP_DIR / filename

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_INFERNO
    )

    cv2.imwrite(
        str(output_path),
        heatmap_color
    )

    return output_path


def save_original(
    image: np.ndarray
):
    
    filename = (
        f"original_{uuid.uuid4()}.jpg"
    )

    output_path = HEATMAP_DIR / filename

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        str(output_path),
        image_bgr
    )

    return output_path