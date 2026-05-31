import os
import cv2
import numpy as np
import time
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

app = FastAPI(title="Silent Face Anti-Spoofing API", description="单目静默活体检测服务", version="1.0.0")

# Initialize models
MODEL_DIR = "./resources/anti_spoof_models"
device_id = 0
model_test = AntiSpoofPredict(device_id)
image_cropper = CropImage()

class LivenessRequest(BaseModel):
    image_base64: str

@app.post("/api/v1/liveness")
async def check_liveness(request: LivenessRequest):
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
                start = time.time()
                prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
                total_test_speed += time.time() - start

            # 预测结果: 0 和 2 代表 Fake（不同维度的攻击如纸张翻拍或屏幕翻拍），1 代表 Real
            label = np.argmax(prediction)
            # 因为有多个模型进行ensemble，所以 / num_models 取平均置信度
            num_models = len([m for m in os.listdir(MODEL_DIR) if m.endswith('.onnx')])
            if num_models == 0:
                raise Exception("未找到 ONNX 模型，请确保预训练模型已成功转换。")
            value = prediction[0][label] / num_models 
            
            is_real = True if label == 1 else False
            
            faces_result.append({
                "is_real": is_real,
                "score": float(value),
                "box": image_bbox
            })
        
        return {
            "code": 200,
            "msg": "success",
            "data": {
                "faces": faces_result,
                "cost_time_sec": round(total_test_speed, 4)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理发生内部错误: {str(e)}")

if __name__ == "__main__":
    print("启动活体检测服务 (Silent Face Anti-Spoofing API)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
