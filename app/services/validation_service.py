from app.services.embedding_similarity_service import (
    compute_similarity
)


MAX_SIMILARITY = 0.85

MIN_MEAN_TOPK = 0.80

MIN_AGREEMENT = 0.80


def build_validation_response(
    *,
    is_valid: bool,
    status: str,
    severity: str,
    title: str,
    message: str,
    similarity_result: dict
):

    return {
        "is_valid": is_valid,
        "status": status,
        "severity": severity,
        "title": title,
        "message": message,
        "metrics": similarity_result["statistics"]
    }


def validate_embedding(
    embedding,
    model_name: str
):

    similarity_result = compute_similarity(
        query_embedding=embedding,
        model_name=model_name,
        top_k=5
    )

    statistics = similarity_result["statistics"]

    max_similarity = statistics["max_similarity"]

    mean_topk = statistics["mean_topk_similarity"]

    agreement = statistics["agreement"]

    is_valid = (

        max_similarity >= VALID_SIMILARITY

        and

        mean_topk >= MIN_MEAN_TOPK

        and

        agreement >= MIN_AGREEMENT

    )

    if is_valid:

        return build_validation_response(

            is_valid=True,
            status="valid",
            severity="success",
            title="",
            message="",
            similarity_result=similarity_result

        )

    return build_validation_response(

        is_valid=False,
        status="invalid",
        severity="warning",
        title="Motif tidak dapat dikenali",
        message=(
            "Gambar belum dapat dipastikan sebagai salah satu "
            "motif batik yang didukung aplikasi. "
            "Pastikan motif terlihat jelas dan gunakan salah satu "
            "motif batik yang tersedia pada aplikasi."
        ),
        similarity_result=similarity_result
        
    )