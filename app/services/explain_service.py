from app.services.inference_service import (
    prepare_input,
    predict_tensor
)

from app.services.gradcam_service import (
    generate_gradcam
)

from app.services.visualization_service import (
    build_visualizations
)

from app.services.explanation_service import (
    generate_explanation
)


def explain_image(
    image_path: str,
    model_name: str
):

    original_image, image_tensor = prepare_input(
        image_path=image_path,
        model_name=model_name
    )

    prediction_result = predict_tensor(
        image_tensor=image_tensor,
        model_name=model_name
    )

    model = prediction_result["model"]

    model_data = prediction_result["model_data"]

    predictions = prediction_result["predictions"]

    pred_index = prediction_result["pred_index"]

    predicted_class = prediction_result["predicted_class"]

    confidence = prediction_result["confidence"]

    gradcam_result = generate_gradcam(

        model=model,

        processed_image=image_tensor,

        pred_index=pred_index,

        backbone_name=model_data["backbone"],

        last_conv_layer_name=model_data["last_conv_layer"]
    )

    heatmap = gradcam_result["raw_heatmap"]

    visualization_result = build_visualizations(
        original_image,
        heatmap
    )

    explanation_result = generate_explanation(
        heatmap=heatmap,
        predicted_class=predicted_class
    )

    return {

        "predicted_class":predicted_class,

        "confidence":confidence,

        "predictions":predictions[0].tolist(),

        "focus_region":explanation_result["focus_region"],

        "focus_percentage":explanation_result["focus_percentage"],

        "attention_distribution":explanation_result["attention_distribution"],

        "active_regions":explanation_result["active_regions"],

        "region_scores":explanation_result["region_scores"],

        "explanation":explanation_result["explanation"],

        "overlay_image":visualization_result["overlay_image"]
    }