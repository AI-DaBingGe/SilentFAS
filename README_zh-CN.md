# 单目静默活体检测 (SilentFAS)

[🇨🇳 中文文档](README_zh-CN.md) | [🇬🇧 English](README.md)

SilentFAS 是一款极速、高精度的单目静默活体检测服务。它仅需一张普通的 RGB 照片，即可精准识别真实人脸与各种作弊攻击（如纸质照片打印、手机/平板屏幕翻拍）。

本项目专为**极高的 CPU 推理性能**打造，使用 **ONNX Runtime** 静态计算图和多模型集成（Ensemble），在不依赖 GPU 的情况下实现了超低的推理延迟，可直接用于线上高并发生产环境。

## 核心特性
- **纯单目静默检测**：无需用户配合做动作（如张嘴、摇头），单张图片直接秒出结果。
- **多人脸支持**：自动检测并判定画面中的所有所有人脸，返回结构化列表。
- **极致的 CPU 性能**：彻底摒弃笨重的 PyTorch 推理，底层替换为 ONNX，普通 CPU 单脸推理耗时低至 `< 15ms`。
- **高并发企业级架构**：内置 `nginx.conf` 模板与多实例启动脚本，完美支持多进程负载均衡，榨干多核算力。
- **Base64 JSON API**：标准的 FastAPI 接口，无缝接入各类业务系统。

## 安装指南

### 环境要求
- Python 3.8+
- Nginx (可选，仅在需要高并发负载均衡时使用)

### 初始化部署
1. 克隆代码仓库:
   ```bash
   git clone https://github.com/AI-DaBingGe/SilentFAS.git
   cd SilentFAS
   ```
2. 安装环境依赖:
   ```bash
   # 强烈建议在国内服务器环境先单独安装纯 CPU 版的 PyTorch，极大地提高安装速度并减小体积：
   python -m pip install torch torchvision --index-url https://mirror.nju.edu.cn/pytorch/whl/cpu --extra-index-url https://mirrors.aliyun.com/pypi/simple/

   # 然后再安装其余依赖：
   pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
   ```
   
## CentOS 7 部署避坑指南 (必看)

如果您在较老的 Linux 系统（如 CentOS 7 / Ubuntu 18.04 以下，即 GLIBC < 2.28）上部署本项目，可能会遇到以下两个经典报错，请按说明解决：

### 1. 报错 `numpy.core.multiarray failed to import`
**原因**：默认安装了最新的 NumPy 2.x，导致与旧版编译的 OpenCV / ONNXRuntime 产生底层 C API 冲突。
**解决**：强制降级 NumPy 到 1.x 版本。
```bash
python -m pip install "numpy<2.0.0" -i https://mirrors.aliyun.com/pypi/simple/
```

### 2. 报错 `Unsupported model IR version: 10, max supported IR version: 9`
**原因**：从 ONNXRuntime `1.17.0` 开始，官方已完全放弃对 CentOS 7 (GLIBC 2.17) 的支持。`pip` 在老系统上最高只能为您安装 `1.16.3` 版本，但该版本最高只支持 IR 版本 9 的模型，而项目自带的模型是 IR 版本 10。
**解决**：无需折腾升级系统，直接运行以下 Python 代码，强行将 ONNX 模型的 IR 版本降级为 9，老版本引擎即可完美读取！

```bash
python -c '\''
import onnx
import glob
print("正在扫描并降级模型版本...")
for p in glob.glob("./resources/anti_spoof_models/*.onnx"):
    m = onnx.load(p)
    if m.ir_version > 9:
        print(f"成功将 {p} 从 IR v{m.ir_version} 降级为 9")
        m.ir_version = 9
        onnx.save(m, p)
print("模型全批降级完成！")
'\''
```

## 使用说明

### 1. 启动服务
启动单实例推理服务：
```bash
python main.py --port 8000
```
*（如果您在 Windows 上需要开启高并发负载均衡，可直接运行 `./start_services.ps1` 启动多个实例，配合 Nginx 分发流量。）*

### 2. 接口调用
**接口地址**：`POST /api/v1/liveness`

**请求参数** (格式: `application/json`)：
请求头 (Headers):
- `X-API-Key: sf_live_platform_A_12345`

```json
{
  "image_base64": "图片的Base64编码字符串..."
}
```

**返回结果**：
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
- `is_real`: `true` 为真人活体，`false` 为假体攻击（照片/屏幕）。
- `score`: 活体置信度得分 (0.0 ~ 1.0)。
- `box`: 人脸在图片中的坐标 `[x, y, 宽, 高]`。

### 3. 全自动验证
项目内置了自动化测试脚本，包含真人与假体翻拍测试图片。请在服务启动后运行：
```bash
python test_api.py
```

## 技术架构
- **人脸检测**：RetinaFace (cv2.dnn Caffe 极速版)
- **防伪推理核心**：MiniFASNet (V1SE 与 V2 双模型集成，ONNX Runtime 加速)
- **Web 服务层**：FastAPI + Uvicorn

## 架构扩展：CPU vs GPU 部署指南

本项目当前架构（ONNX Runtime CPU 引擎 + Nginx 多进程负载均衡）是专为 **纯 CPU 服务器** 打造的终极高并发方案，能充分压榨多核算力。

**如果想使用 GPU 加速：**
代码兼容性极高，只需极小改动即可支持 Nvidia GPU：
1. 安装依赖：`pip install onnxruntime-gpu`
2. 修改 `src/anti_spoof_predict.py`，将 `['CPUExecutionProvider']` 替换为 `['CUDAExecutionProvider']`。单脸推理时间将下降至毫秒级。

**生产环境 GPU 极高并发建议：**
虽然本架构可无缝切换至 GPU，但如果您的目标是利用 A100/T4 等高端显卡实现 5000+ QPS 的极致并发，传统的 `Nginx + FastAPI` 处理单张图片（Batch Size = 1）会造成算力浪费。对于真正的工业级 GPU 集群，推荐基于本项目的 ONNX 模型，改用 **Nvidia Triton Inference Server** 结合**动态批处理 (Dynamic Batching)** 技术，打包矩阵并发推理，以获得数十倍的吞吐量跃升。

## 协议
MIT License
