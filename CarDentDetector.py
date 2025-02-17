from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import json
import math
import cvzone
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from datetime import datetime

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Directory for uploads
app.config['LIVE_DETECTION_FOLDER'] = 'static/live_detections/'  # Directory for live feed detections
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['LIVE_DETECTION_FOLDER'], exist_ok=True)

# Load YOLO model
MODEL_PATH = "Weights/best.pt"
model = YOLO(MODEL_PATH)

# Define class labels
class_labels = [
    'Bodypanel-Dent', 'Front-Windscreen-Damage', 'Headlight-Damage',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'boot-dent',
    'doorouter-dent', 'fender-dent', 'front-bumper-dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

# Load or create detected dents JSON file
DETECTED_DENTS_FILE = "detected_dents.json"
if os.path.exists(DETECTED_DENTS_FILE):
    with open(DETECTED_DENTS_FILE, "r") as f:
        detected_dents = json.load(f)
else:
    detected_dents = []

# Save detected dents to JSON file
def save_dents():
    with open(DETECTED_DENTS_FILE, "w") as f:
        json.dump(detected_dents, f, indent=4)

# Check if file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Process image
def process_image(file_path):
    img = cv2.imread(file_path)
    results = model(img)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(box.conf[0].item(), 2)
            cls = int(box.cls[0])

            if conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), t=2)
                cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

                detected_dents.append({
                    'type': class_labels[cls],
                    'confidence': conf,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'file_path': file_path
                })
                save_dents()

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{os.path.basename(file_path)}')
    cv2.imwrite(output_path, img)
    return output_path

# Process video
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{os.path.basename(file_path)}')

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(box.conf[0].item(), 2)
                cls = int(box.cls[0])

                if conf > 0.3:
                    cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), t=2)
                    cvzone.putTextRect(frame, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

                    detected_dents.append({
                        'type': class_labels[cls],
                        'confidence': conf,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'file_path': file_path
                    })
                    save_dents()

        out.write(frame)

    cap.release()
    out.release()
    return output_path

# Live feed generator
def generate_frames():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(box.conf[0].item(), 2)
                cls = int(box.cls[0])

                if conf > 0.3:
                    cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), t=2)
                    cvzone.putTextRect(frame, f'{class_labels[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

                    detected_dents.append({
                        'type': class_labels[cls],
                        'confidence': conf,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'file_path': 'live_feed'
                    })
                    save_dents()

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        output_path = process_image(file_path) if filename.split('.')[-1].lower() in ['jpg', 'jpeg', 'png'] else process_video(file_path)

        return render_template('result.html', output_file=output_path, file_type=filename.split('.')[-1].lower(), detected_dents=detected_dents)

@app.route('/live_feed')
def live_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_dents')
def show_detected_dents():
    return render_template('detected_dents.html', detected_dents=detected_dents)

if __name__ == '__main__':
    app.run(debug=True)
