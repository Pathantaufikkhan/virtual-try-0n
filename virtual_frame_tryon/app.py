from flask import Flask, render_template, request
import cv2
import dlib
import numpy as np
import os

app = Flask(__name__)

# Load Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")

# Load frames
frames_path = "./frame"
frames = [
    cv2.imread(os.path.join(frames_path, f), cv2.IMREAD_UNCHANGED)
    for f in os.listdir(frames_path)
    if f.endswith('.png')
]

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Read uploaded image
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Process image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
                glasses_frame, position = resize_frame(frames[0], landmarks)
                overlay_frame(img, glasses_frame, position)

            _, img_encoded = cv2.imencode('.jpg', img)
            return img_encoded.tobytes()

    return render_template("index.html")

# Utility functions (resize_frame, overlay_frame)
def resize_frame(frame, landmarks):
    # (Same resize logic as your earlier script)
    pass

def overlay_frame(image, frame, position):
    # (Same overlay logic as your earlier script)
    pass

if __name__ == "__main__":
    app.run(debug=True)
