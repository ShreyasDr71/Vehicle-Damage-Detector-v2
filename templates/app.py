from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import torch
from pathlib import Path

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLOv8 model
MODEL_PATH = "Weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)  # Replace with YOLOv8 loading if needed

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Process image
def process_image(file_path):
    results = model(file_path)
    output_path = file_path.replace('uploads/', 'uploads/output_')
    results.save(save_dir=Path(output_path).parent)
    return output_path

# Process video
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_path = file_path.replace('uploads/', 'uploads/output_')

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        annotated_frame = results.render()[0]
        out.write(annotated_frame)

    cap.release()
    out.release()
    return output_path

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check file type and process accordingly
        if filename.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
            output_path = process_image(file_path)
        elif filename.split('.')[-1].lower() in ['mp4']:
            output_path = process_video(file_path)
        else:
            return "Unsupported file type."

        return redirect(url_for('uploaded_file', filename=output_path.split('/')[-1]))

    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
