# app/api.py

from flask import Flask, request, jsonify
import numpy as np
import cv2
from .detection_logic import detect_faces

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = detect_faces(image)
    return jsonify({"results": results})
