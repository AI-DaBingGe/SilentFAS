# Silent Face Anti-Spoofing (SilentFAS)

[🇨🇳 中文文档](README_zh-CN.md) | [🇬🇧 English](README.md)

SilentFAS is an ultra-fast, high-accuracy Face Anti-Spoofing (Liveness Detection) service. It accurately identifies real faces versus spoofing attacks (e.g., printed photos, screen replays) using only a single RGB image.

Designed for extreme CPU performance, this project utilizes **ONNX Runtime** and **Dynamic Ensemble Models** to achieve blazingly fast inference speeds, making it highly suitable for high-concurrency production environments without requiring GPUs.

## Features
- **Monocular Liveness Detection**: Requires only a single standard RGB image.
- **Multi-Face Support**: Automatically detects and processes all faces within an image.
- **Extreme CPU Performance**: Replaced PyTorch backend with ONNX Runtime static graphs, dropping inference latency to `< 15ms` per face on standard CPUs.
- **High Concurrency Ready**: Includes `nginx.conf` and startup scripts to natively support multi-instance load balancing.
- **Base64 JSON API**: Standardized FastAPI interface for seamless integration.

## Installation

### Prerequisites
- Python 3.8+
- Nginx (Optional, for load balancing)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/AI-DaBingGe/SilentFAS.git
   cd SilentFAS
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Start the Service
You can start a single FastAPI instance:
```bash
python main.py --port 8000
```
*(Windows users can also use `./start_services.ps1` to launch multiple instances for Nginx load balancing).*

### 2. API Endpoint
**POST** `/api/v1/liveness`

**Request Body** (`application/json`):
```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAE..."
}
```

**Response**:
```json
{
    "code": 200,
    "msg": "success",
    "data": {
        "faces": [
            {
                "is_real": true,
                "score": 0.9928,
                "box": [350, 236, 331, 338]
            }
        ],
        "cost_time_sec": 0.0125
    }
}
```

### 3. Automated Testing
Run the built-in test suite to verify the installation against real and fake sample images:
```bash
python test_api.py
```

## Architecture
- **Detector**: RetinaFace (cv2.dnn Caffe implementation)
- **Anti-Spoofing Engine**: MiniFASNet (Ensemble of V1SE and V2, compiled to ONNX)
- **Serving**: FastAPI + Uvicorn

## License
MIT License
