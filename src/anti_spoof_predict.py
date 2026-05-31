# -*- coding: utf-8 -*-

import os
import cv2
import math
import numpy as np
import onnxruntime as ort

from src.data_io import transform as trans
from src.utility import parse_model_name

class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bboxes(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        
        bboxes = []
        if len(out.shape) == 1 and out.shape[0] == 7:
            out = np.expand_dims(out, axis=0)
        elif len(out.shape) == 0 or out.shape[0] == 0:
            return bboxes
            
        for i in range(out.shape[0]):
            conf = out[i, 2]
            if conf > self.detector_confidence:
                left, top, right, bottom = out[i, 3]*width, out[i, 4]*height, \
                                           out[i, 5]*width, out[i, 6]*height
                w, h = int(right-left+1), int(bottom-top+1)
                if w > 0 and h > 0:
                    bboxes.append([int(left), int(top), w, h])
        return bboxes

class AntiSpoofPredict(Detection):
    def __init__(self, device_id=0):
        super(AntiSpoofPredict, self).__init__()
        self.providers = ['CPUExecutionProvider']
        self.sessions = {}

    def _load_model(self, model_path):
        if model_path not in self.sessions:
            sess_options = ort.SessionOptions()
            # CPU performance tuning
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sessions[model_path] = ort.InferenceSession(model_path, sess_options, providers=self.providers)
        return self.sessions[model_path]

    def predict(self, img, model_path):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img_np = img.unsqueeze(0).numpy()
        
        session = self._load_model(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        result = session.run([output_name], {input_name: img_np})[0]
        
        exp_result = np.exp(result - np.max(result))
        softmax_result = exp_result / exp_result.sum(axis=1, keepdims=True)
        return softmax_result
