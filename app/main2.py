import os
import torch
import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

# ==========================================================
# 1. CRITICAL PYTORCH 2.6 FIX
# This bypasses the UnpicklingError you were receiving
# ==========================================================
import torch.serialization
torch.serialization.safe_globals = lambda *args, **kwargs: None 

# ==========================================================
# 2. DIRECTORY & PATH SETUP
# Ensures Flask finds 'templates' and 'static' correctly
# ==========================================================
# Get the absolute path of the directory where main2.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory (one level up from 'app')
ROOT_DIR = os.path.dirname(BASE_DIR)

app = Flask(__name__, 
            template_folder=os.path.join(ROOT_DIR, 'templates'),
            static_folder=os.path.join(ROOT_DIR, 'static'))

# Path to your custom weights file
MODEL_PATH = os.path.join(ROOT_DIR, "model", "yolov8_custom.pt")

# ==========================================================
# 3. LOAD THE MODEL
# ==========================================================
try:
    # First, try loading your custom weights
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded custom model: {MODEL_PATH}")
    else:
        # Fallback if your custom model is still missing or 0KB
        print("Warning: Custom model is missing or 0KB. Loading yolov8n.pt for testing.")
        model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = YOLO("yolov8n.pt")

# ==========================================================
# 4. VIDEO STREAMING LOGIC
# ==========================================================
def generate_frames():
    # 0 is the default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLOv8 object detection
        results = model(frame, conf=0.5)
        
        # Draw the detection boxes on the frame
        annotated_frame = results[0].plot()

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        # Yield the output in the format Flask expects for video
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================================
# 5. ROUTES
# ==========================================================
@app.route('/')
def index():
    """Main Home Page"""
    return render_template('index.html')

@app.route('/live_detect')
def live_detect():
    """Page for live detection"""
    return render_template('live_detect.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route used in <img> tags in HTML"""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================================
# 6. RUN THE APP
# ==========================================================
if __name__ == "__main__":
    # threaded=True is important for video streaming
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)