from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "batik_betawi",
    "batik_bokor_kencono",
    "batik_buketan",
    "batik_dayak",
    "batik_jlamprang",
    "batik_kawung",
    "batik_liong",
    "batik_mega_mendung",
    "batik_parang",
    "batik_sekarjagad",
    "batik_sidoluhur",
    "batik_sidomukti",
    "batik_sidomulyo",
    "batik_singa_barong",
    "batik_srikaton",
    "batik_tribusono",
    "batik_tujuh_rupa",
    "batik_tuntrum",
    "batik_wahyu_tumurun",
    "batik_wirasat",
]

MODEL_CONFIGS = {
    "mobilenetv2": {
        "display_name": "MobileNetV2",
        "model_path": BASE_DIR / "artifacts/models/MobileNetV2.keras",
        "backbone": "mobilenetv2_1.00_224",
        "last_conv_layer": "out_relu",
    },
    "resnet50": {
        "display_name": "ResNet50",
        "model_path": BASE_DIR / "artifacts/models/ResNet50.keras",
        "backbone": "resnet50",
        "last_conv_layer": "conv5_block3_out",
    },
}

REGION_LABELS = {
    "top_left": "kiri atas",
    "top_center": "tengah atas",
    "top_right": "kanan atas",

    "middle_left": "kiri tengah",
    "center": "tengah",
    "middle_right": "kanan tengah",

    "bottom_left": "kiri bawah",
    "bottom_center": "tengah bawah",
    "bottom_right": "kanan bawah"
}