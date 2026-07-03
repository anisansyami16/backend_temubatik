import base64
import cv2
import numpy as np


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

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    return cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_INFERNO
    )


def build_heatmap(
    heatmap: np.ndarray,
    target_shape: tuple
):

    heatmap = resize_heatmap(
        heatmap,
        target_shape
    )

    threshold = 0.15

    filtered = np.zeros_like(
        heatmap
    )

    filtered[
        heatmap >= threshold
    ] = heatmap[
        heatmap >= threshold
    ]

    return apply_colormap(
        filtered
    )


def build_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
):

    heatmap_color = build_heatmap(
        heatmap,
        original_image.shape
    )

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


def encode_image(
    image: np.ndarray
):

    success, buffer = cv2.imencode(
        ".jpg",
        image
    )

    if not success:

        raise ValueError(
            "Failed to encode image."
        )

    return base64.b64encode(
        buffer
    ).decode("utf-8")


def build_visualizations(
    original_image: np.ndarray,
    heatmap: np.ndarray
):

    overlay_image = build_overlay(
        original_image,
        heatmap
    )

    return {

        "overlay_image":
            encode_image(overlay_image)

    }