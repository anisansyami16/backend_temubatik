import numpy as np

from app.core.constants import CLASS_NAMES

from app.services.gradcam_service import (
    generate_gradcam
)

from app.services.visualization_service import (
    build_overlay,
    save_overlay
)

from app.services.explanation_service import (
    generate_explanation
)


def generate_full_explanation(
    model,
    model_data,
    image,
    image_tensor,
    predictions
):
  
    pred_index = np.argmax(
        predictions[0]
    )

    predicted_class = CLASS_NAMES[pred_index]

    confidence = float(
        predictions[0][pred_index]
    )

    gradcam_result = (
        generate_gradcam(
            model=model,
            processed_image=image_tensor,
            pred_index=pred_index,
            backbone_name=model_data["backbone"],
            last_conv_layer_name=model_data["last_conv_layer"]
        )
    )

    heatmap = gradcam_result["raw_heatmap"]

    overlay = build_overlay(
        original_image=image,
        heatmap=heatmap
    )

    overlay_path = save_overlay(
        overlay
    )

    explanation_result = (
        generate_explanation(
            heatmap=heatmap,
            predicted_class=predicted_class
        )
    )

    print(explanation_result)

    return {
        
        "predicted_class":predicted_class,

        "confidence":confidence,

        "focus_region":explanation_result["focus_region"],

        "focus_percentage":explanation_result["focus_percentage"],

        "peak_activation":explanation_result["peak_activation"],

        "attention_distribution":explanation_result["attention_distribution"],

        "active_regions":explanation_result["active_regions"],

        "region_scores":explanation_result["region_scores"],

        "explanation":explanation_result["explanation"],

        "overlay_image":str(overlay_path)

    }