from flask import Flask, request, jsonify
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize models
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)

    faces = face_app.get(img)
    # (Do recognition logic here and return JSON with names.)

    return jsonify({"faces_detected": len(faces)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
