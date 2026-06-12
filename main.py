import os
import cv2
import numpy as np
import time
import base64
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import uvicorn

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.image_enhance import check_face_quality, adaptive_gamma_correction, detect_moire_fft, apply_clahe_contrast

app = FastAPI(title="Silent Face Anti-Spoofing API", description="单目静默活体检测服务", version="1.0.0")

# --- Security: API Key Authorization ---
API_KEY_NAME = "X-API-Key"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 默认白名单 Token 配置字典 (未来可扩展为查询 Redis 或 MySQL)
VALID_API_KEYS = {
    "sf_live_platform_A_12345": "合作平台 A",
    "sf_live_platform_B_67890": "合作平台 B"
}

async def get_api_key(api_key: str = Security(api_key_header_auth)):
    if api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效或缺失的 API Key 授权 (X-API-Key)",
    )
# ---------------------------------------

# Initialize models
MODEL_DIR = "./resources/anti_spoof_models"
device_id = 0
model_test = AntiSpoofPredict(device_id)
image_cropper = CropImage()

class LivenessRequest(BaseModel):
    image_base64: str

@app.post("/api/v1/liveness")
async def check_liveness(request: LivenessRequest, api_key: str = Security(get_api_key)):
    """
    接收单张图片的 Base64 编码，进行活体检测。
    返回 is_real (布尔值) 以及 score (概率得分)。
    """
    try:
        base64_data = request.image_base64
        # 如果包含头部信息 (如 data:image/jpeg;base64,)，将其剔除
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        image_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图片，请检查 Base64 编码是否有效。")
            
        # --------------------------------------------------------
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 解码或图片读取失败: {str(e)}")

    try:
        # Detect Face
        image_bboxes = model_test.get_bboxes(image)
        if image_bboxes is None or len(image_bboxes) == 0:
            return {"code": 400, "msg": "未检测到人脸，请提供清晰的人脸照片。", "data": {"faces": []}}
            
        faces_result = []
        total_test_speed = 0
        
        for image_bbox in image_bboxes:
            prediction = np.zeros((1, 3))
            
            checked_quality = False
            brightness = 255.0
            
            # 遍历所有模型进行 Ensemble(集成推理)
            for model_name in os.listdir(MODEL_DIR):
                if not model_name.endswith('.onnx'):
                    continue
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)
                
                # --- 多维交叉验证预处理 ---
                if not checked_quality:
                    brightness = check_face_quality(img)
                    checked_quality = True
                
                img = adaptive_gamma_correction(img, brightness)
                # --------------------------
                
                start = time.time()
                prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
                total_test_speed += time.time() - start

            # 预测结果处理
            num_models = len([m for m in os.listdir(MODEL_DIR) if m.endswith('.onnx')])
            if num_models == 0:
                raise Exception("未找到 ONNX 模型，请确保预训练模型已成功转换。")
                
            real_score = prediction[0][1] / num_models
            fake_paper_score = prediction[0][0] / num_models
            fake_screen_score = prediction[0][2] / num_models
            
            # 多维综合判定逻辑
            if real_score > 0.85:
                is_real = True
            elif real_score > 0.3:
                # 处于模糊地带，使用物理摩尔纹检测作为第二防线（使用当前裁剪图即可）
                has_moire = detect_moire_fft(img)
                if has_moire:
                    is_real = False
                    real_score = min(real_score, 0.4) # 惩罚得分，防止混淆
                else:
                    # 如果不是极暗环境或者摩尔纹不明显，可以相信模型倾向
                    is_real = True if real_score > 0.5 else False
            else:
                is_real = False
                
            # 若屏幕翻拍概率直接过半，强行拦截
            if fake_screen_score > 0.5:
                is_real = False
                
            value = real_score
            
            faces_result.append({
                "is_real": is_real,
                "score": float(value),
                "box": image_bbox
            })
        
        response_data = {
            "code": 200,
            "msg": "success",
            "data": {
                "faces": faces_result,
                "cost_time_sec": round(total_test_speed, 4)
            }
        }
        print(f"INFO:     [Response Data] {response_data}", flush=True)
        return response_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理发生内部错误: {str(e)}")

if __name__ == "__main__":
    print("启动活体检测服务 (Silent Face Anti-Spoofing API)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
