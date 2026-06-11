import cv2
import numpy as np
from fastapi import HTTPException

def check_face_quality(face_img, min_brightness=30):
    """
    检查图片质量，如果图片太暗则直接抛出 400 异常
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < min_brightness:
        raise HTTPException(
            status_code=400, 
            detail=f"环境光线过暗 (亮度: {mean_brightness:.1f})，请在明亮处重试。"
        )
    return mean_brightness

def adaptive_gamma_correction(face_img, brightness, threshold=80, gamma=1.8):
    """
    自适应 Gamma 校正：如果亮度低于阈值，则使用 Gamma 曲线提亮
    """
    if brightness < threshold:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(face_img, table)
    return face_img

def apply_clahe_contrast(face_img):
    """
    使用 CLAHE (限制对比度自适应直方图均衡化) 增强图像立体感和对比度
    这对非常扁平的假脸或劣质摄像头图像有极大的“立体感还原”效果。
    """
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def detect_moire_fft(face_img):
    """
    使用傅里叶变换检测高频摩尔纹。
    返回 bool 表示是否检测到摩尔纹。
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # 傅里叶变换
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 获取图像尺寸并屏蔽低频中心部分
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # 屏蔽中心低频区域 (半径 30 像素)
    r = min(30, rows//4, cols//4) # 动态调整半径防止越界
    mask = np.ones((rows, cols), np.uint8)
    if crow > r and ccol > r:
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0
    
    high_freq_magnitude = magnitude_spectrum * mask
    
    max_val = np.max(high_freq_magnitude)
    if max_val < 10: 
        return False
        
    # 统计高频突出峰值数量。摩尔纹会在高频区域形成离散的亮点
    peaks = np.sum(high_freq_magnitude > (max_val * 0.8))
    
    if 2 < peaks < 100:
        return True
        
    return False
