# Silent Face Anti-Spoofing API (单目静默活体检测服务)

本项目提供了一个基于深度学习的单目静默活体检测 HTTP API 服务。它旨在替代如虹软 (ArcSoft) 等商业 SDK 的活体检测功能。
它能够仅仅通过一张图片（无需动作配合），识别图片中的人脸是**真人**还是**假体**（如手机翻拍、屏幕翻拍、打印照片等）。

核心算法基于开源社区优秀的 [MiniVision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)。
本服务使用 FastAPI 进行了高度封装，并去除了冗余的训练代码，提供了开箱即用的推理接口，原生支持跨平台 (Windows/Linux/Mac) 及 CPU/GPU 混合调度。

## 特性
- **轻量级**：使用轻量级网络结构 (MiniFASNet)，在 CPU 上即可实现毫秒级响应。
- **高精度**：融合多个尺度的模型进行集成推理 (Ensemble)，有效对抗各种维度的攻击。
- **易于集成**：标准的 RESTful API (FastAPI)，任何语言均可通过 HTTP 调用。

## 环境安装

1. **克隆项目并进入目录**
2. **安装依赖** (建议使用 Python 3.8+ 虚拟环境)：
   ```bash
   pip install -r requirements.txt
   ```
   > 提示：`torch` 和 `torchvision` 将默认安装最新版本，兼容 CPU。如果您有 NVIDIA 显卡并希望使用 GPU 加速，请前往 [PyTorch官网](https://pytorch.org/) 安装带 CUDA 支持的版本，服务会自动识别并启用 GPU。

## 启动服务

```bash
python main.py
```
服务默认在 `http://0.0.0.0:8000` 启动。
您可以访问 `http://127.0.0.0:8000/docs` 查看 Swagger 交互式接口文档。

## API 调用示例

### `POST /api/v1/liveness`

**请求参数 (application/json):**
- `image_base64`: 待检测的图片 Base64 编码字符串（支持带有或不带 `data:image/jpeg;base64,` 头部）。

**Python 请求示例:**
```python
import requests
import base64

# 读取图片并转为 base64
with open('./images/sample/image_T1.jpg', 'rb') as f:
    base64_data = base64.b64encode(f.read()).decode('utf-8')

url = "http://127.0.0.1:8000/api/v1/liveness"
payload = {"image_base64": base64_data}
response = requests.post(url, json=payload)
print(response.json())
```

**成功响应 (JSON):**
```json
{
    "code": 200,
    "msg": "success",
    "data": {
        "is_real": true,
        "score": 0.985,
        "box": [100, 80, 200, 200],
        "cost_time_sec": 0.045
    }
}
```
- `is_real`: `true` 表示判断为活体（真人），`false` 表示假体（翻拍）。
- `score`: 为活体的置信度得分（0~1）。
- `box`: 画面中检测到的人脸坐标 `[x, y, width, height]`。

## 目录结构
- `main.py`: FastAPI 服务入口及推理逻辑封装。
- `src/`: 核心模型网络结构及图像预处理逻辑。
- `resources/`: 存放预训练的人脸检测模型及活体分类模型。
- `images/`: 测试用的示例图片。

## 开源协议
核心算法基于原仓库的开源协议，本项目额外封装的 API 部分完全开源。
