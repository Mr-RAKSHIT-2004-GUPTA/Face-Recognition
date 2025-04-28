# frontend/gradio_app.py

import gradio as gr
import requests
import cv2
import numpy as np

API_URL = "https://your-backend-app.onrender.com/predict"

def recognize_face(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    response = requests.post(API_URL, files={"file": img_encoded.tobytes()})
    if response.ok:
        results = response.json()["results"]
        return str(results)
    else:
        return "Failed to detect face."

gr.Interface(fn=recognize_face, inputs="image", outputs="text").launch()
