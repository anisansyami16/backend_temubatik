# temu-batik-back-end

## Requirements
- Python 3.10.x
- TensorFlow 2.21.0
- Keras 3.12.1

---

## Setup Environment

### 1.Create Virtual Environment

Jika sudah menggunakan Python 3.10 secara global:

`python -m venv venv`

Jika memiliki multiple Python version di sistem:

`py -3.10 -m venv venv`

---

### 2.Activate Virtual Environment

Windows CMD:

`venv\Scripts\activate`

PowerShell:

`venv\Scripts\Activate.ps1`

Jika berhasil, terminal akan menampilkan `(venv)` di awal path.

---

### 3.Upgrade pip

`python -m pip install --upgrade pip`

---

### 4.Install Dependencies

`pip install -r requirements-lock.txt`

---

## 5.Model Files

Pastikan file model sudah tersedia sebelum menjalankan backend.

Contoh struktur:

```
app/
└── artifacts/
    └── models/
        ├── MobileNetV2.keras
        └── ResNet50.keras
```

---

## Run Development Server

`python -m uvicorn app.main:app --reload`

Expected output:

Starting Temu Batik API...
Loading model: mobilenetv2
Successfully loaded: mobilenetv2
Loading model: resnet50
Successfully loaded: resnet50
All models loaded successfully

Jika Gagal pastikan:
- python sudah 3.10.x
- tensorflow 2.21.0
- keras 3.12.1
- file model tersedia di direktori app/artifacts/models

---

## API Documentation

Swagger UI:

`http://127.0.0.1:8000/docs`


.
├── .dockerignore
├── .env
├── .gitignore
├── app
│   ├── api
│   │   ├── router.py
│   │   └── routes
│   │       ├── debug.py
│   │       ├── explain.py
│   │       ├── health.py
│   │       ├── predict.py
│   │       └── root.py
│   ├── artifacts
│   │   └── models
│   │       ├── MobileNetV2.keras
│   │       └── ResNet50.keras
│   ├── core
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── logger.py
│   │   └── model_registry.py
│   ├── main.py
│   ├── models
│   │   └── response
│   │       └── .gitkeep
│   ├── schemas
│   │   ├── explanation.py
│   │   └── prediction.py
│   ├── services
│   │   ├── explanation_service.py
│   │   ├── gradcam_service.py
│   │   ├── inference_service.py
│   │   ├── lime_service.py
│   │   └── visualization_service.py
│   ├── temp
│   │   ├── .gitkeep
│   │   ├── heatmaps
│   │   │   └── .gitkeep
│   │   └── uploads
│   │       ├── .gitkeep
│   │       └── f8d74ea3-1980-4abf-88aa-99c010ece4c3.jpg
│   └── utils
│       ├── file.py
│       ├── image.py
│       ├── preprocessing.py
│       ├── tensor.py
│       └── tree.py
├── Dockerfile
├── README.md
├── requirements-lock.txt
└── requirements.txt