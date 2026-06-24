import cv2
import numpy as np

from app.core.constants import (
    IMAGE_SIZE,
    REGION_LABELS
)


def get_focus_region(
    heatmap: np.ndarray
):
  
    height, width = heatmap.shape

    h = height // 3
    w = width // 3

    regions = {
        "top_left":heatmap[0:h, 0:w],

        "top_center":heatmap[0:h, w:2*w],

        "top_right":heatmap[0:h, 2*w:],

        "middle_left":heatmap[h:2*h, 0:w],

        "center":heatmap[h:2*h, w:2*w],

        "middle_right":heatmap[h:2*h, 2*w:],

        "bottom_left":heatmap[2*h:, 0:w],

        "bottom_center":heatmap[2*h:, w:2*w],

        "bottom_right":heatmap[2*h:, 2*w:]
    }

    scores = {
        region: float(np.mean(values))
        for region, values
        in regions.items()
    }

    focus_region = max(
        scores,
        key=scores.get
    )

    return focus_region, scores


def count_active_regions(
    heatmap: np.ndarray,
    threshold: float = 0.5
):

    height, width = heatmap.shape

    h = height // 3
    w = width // 3

    regions = {
        "top_left": heatmap[0:h, 0:w],

        "top_center": heatmap[0:h, w:2*w],

        "top_right": heatmap[0:h, 2*w:],

        "middle_left": heatmap[h:2*h, 0:w],

        "center": heatmap[h:2*h, w:2*w],

        "middle_right": heatmap[h:2*h, 2*w:],

        "bottom_left": heatmap[2*h:, 0:w],

        "bottom_center": heatmap[2*h:, w:2*w],

        "bottom_right": heatmap[2*h:, 2*w:]
    }

    active = 0

    for region in regions.values():

        region_score = np.mean(region)

        if region_score >= threshold:

            active += 1

    return active


def get_focus_percentage(
    heatmap: np.ndarray,
    threshold: float = 0.5
):
    
    active_pixels = np.sum(
        heatmap >= threshold
    )

    total_pixels = heatmap.size

    return round(
        active_pixels / total_pixels * 100,
        2
    )


def get_peak_activation(
    heatmap: np.ndarray
):
    
    return round(
        float(np.max(heatmap)),
        4
    )


def get_region_description(
    focus_region: str
):

    return REGION_LABELS.get(
        focus_region,
        focus_region
    )


def get_attention_distribution(
    focus_percentage: float
):
    
    if focus_percentage < 20:
        return "highly_localized"

    if focus_percentage < 50:
        return "localized"

    if focus_percentage < 75:
        return "moderate"

    return "diffuse"



def get_distribution_description(
    attention_distribution: str
):
    
    descriptions = {

        "highly_localized":
            (
                "Model memusatkan perhatian pada "
                "area yang sangat spesifik."
            ),

        "localized":
            (
                "Model lebih banyak "
                "berfokus pada sebagian kecil "
                "area gambar."
            ),

        "moderate":
            (
                "Model memperhatikan beberapa "
                "bagian motif yang saling terkait."
            ),

        "diffuse":
            (
                "Model memanfaatkan area motif "
                "yang tersebar luas pada gambar."
            )
    }

    return descriptions.get(
        attention_distribution,
        ""
    )


def get_active_region_description(
    active_regions: int
):
    
    if active_regions == 1:

        return (
            "Keputusan model terutama "
            "ditentukan oleh satu area dominan."
        )

    if active_regions <= 3:

        return (
            "Beberapa area motif "
            "berkontribusi terhadap prediksi."
        )

    return (
        "Prediksi dipengaruhi oleh "
        "kombinasi beberapa area motif."
    )


def build_explanation(
    predicted_class,
    focus_region,
    focus_percentage,
    peak_activation,
    attention_distribution,
    active_regions
):
    
    region_text = get_region_description(focus_region)

    distribution_text = get_distribution_description(attention_distribution)

    active_region_text = get_active_region_description(active_regions)

    return (

        f"Model memberikan perhatian "
        f"terbesar pada area {region_text}. "

        f"{distribution_text} "

        f"{active_region_text} "

        f"Sekitar {focus_percentage}% area "
        f"heatmap memberikan kontribusi "
        f"terhadap prediksi motif "
        f"{predicted_class}. "

        f"Tingkat aktivasi maksimum "
        f"mencapai {peak_activation}."
    )


def generate_explanation(
    heatmap: np.ndarray,
    predicted_class: str
):
    
    heatmap = cv2.resize(heatmap, IMAGE_SIZE)

    focus_region, region_scores = get_focus_region(heatmap)

    focus_percentage = get_focus_percentage(heatmap)

    peak_activation = get_peak_activation(heatmap)

    attention_distribution = get_attention_distribution(focus_percentage)

    active_regions = count_active_regions(heatmap)

    explanation = build_explanation(
        predicted_class,
        focus_region,
        focus_percentage,
        peak_activation,
        attention_distribution,
        active_regions
    )

    return {
        "focus_region": str(focus_region),

        "focus_percentage": float(focus_percentage),

        "peak_activation": float(peak_activation),

        "attention_distribution": str(attention_distribution),

        "active_regions": int(active_regions),

        "region_scores":region_scores,

        "explanation": str(explanation)
    }