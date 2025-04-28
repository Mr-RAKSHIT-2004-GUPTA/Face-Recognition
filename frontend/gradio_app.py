import gradio as gr
import requests

def recognize_face(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    response = requests.post("http://your-server-ip:5000/predict", files={"file": img_encoded.tobytes()})
    return response.json()

gr.Interface(fn=recognize_face, inputs="image", outputs="text").launch()
