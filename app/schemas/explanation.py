from pydantic import BaseModel


class ExplanationResponse(
    BaseModel
):

    predicted_class: str

    confidence: float

    focus_region: str

    focus_percentage: float

    peak_activation: float

    attention_distribution: str

    active_regions: int

    explanation: str

    overlay_image: str