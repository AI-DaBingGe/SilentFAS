import os
import cv2
import numpy as np
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
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

@app.post("/api/v1/liveness")
async def check_liveness(file: UploadFile = File(...)):
    """
    接收单张图片，进行活体检测。
    返回 is_real (布尔值) 以及 score (概率得分)。
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="只支持 PNG 和 JPG 格式的图片。")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="无法解析图片，请检查图片是否损坏。")

    try:
        # Detect Face
        image_bbox = model_test.get_bbox(image)
        if image_bbox is None or len(image_bbox) < 4 or image_bbox[2] == 0 or image_bbox[3] == 0:
            return {"code": 400, "msg": "未检测到人脸，请提供清晰的人脸照片。", "data": None}
            
        prediction = np.zeros((1, 3))
        test_speed = 0
        
        # 遍历所有模型进行 Ensemble(集成推理)
        for model_name in os.listdir(MODEL_DIR):
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
            test_speed += time.time() - start

        # 预测结果: 0 和 2 代表 Fake（不同维度的攻击如纸张翻拍或屏幕翻拍），1 代表 Real
        label = np.argmax(prediction)
        # 因为有2个模型（默认为2个模型进行ensemble），所以 / 2 取平均置信度
        num_models = len(os.listdir(MODEL_DIR))
        value = prediction[0][label] / num_models 
        
        is_real = True if label == 1 else False
        
        return {
            "code": 200,
            "msg": "success",
            "data": {
                "is_real": is_real,
                "score": float(value),
                "box": image_bbox,
                "cost_time_sec": round(test_speed, 4)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理发生内部错误: {str(e)}")

if __name__ == "__main__":
    print("启动活体检测服务 (Silent Face Anti-Spoofing API)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
