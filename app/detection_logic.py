# app/detection_logic.py

import cv2
import numpy as np
import insightface

# Load the models once globally
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_size=(640, 640))

def detect_faces(image):
    # image comes as OpenCV format
    faces = model.get(image)
    
    if len(faces) == 0:
        return "No face detected"

    result = []
    for face in faces:
        box = face.bbox.astype(int)
        name = "Unknown"
        # (Here you can add matching logic if you have known faces embeddings)
        result.append({"bbox": box.tolist(), "name": name})

    return result
