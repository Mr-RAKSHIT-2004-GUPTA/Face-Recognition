import cv2
import os
import insightface
import numpy as np
from numpy.linalg import norm

# Load the face detector and embedding model
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) 
app.prepare(ctx_id=0, det_size=(640, 640))


# Database of known faces
known_faces = {}

# Function to register faces
def register_faces(known_faces_folder):
    for filename in os.listdir(known_faces_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(known_faces_folder, filename)
            img = cv2.imread(path)
            faces = app.get(img)
            if len(faces) == 0:
                print(f"[WARNING] No face detected in {filename}")
                continue
            embedding = faces[0].embedding
            name = os.path.splitext(filename)[0]  # remove .jpg extension
            known_faces[name] = embedding
            print(f"[INFO] Registered {name}")

# Register faces
register_faces("known_faces")

# Function to compare two embeddings
def compare_embeddings(emb1, emb2, threshold=0.5):
    cos_sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return cos_sim > (1 - threshold)

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        embedding = face.embedding

        name = "Unknown"
        best_score = -1
        for registered_name, registered_emb in known_faces.items():
            cos_sim = np.dot(embedding, registered_emb) / (norm(embedding) * norm(registered_emb))
            if cos_sim > best_score:
                best_score = cos_sim
                name = registered_name

        if best_score < 0.3:  # Adjust threshold here
            name = "Unknown"

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({best_score:.2f})", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
