from flask import Flask, request, jsonify
import cv2
import numpy as np
import cvzone
from ultralytics import YOLO
from pathlib import Path
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --------------------------
# Setup paths and model
# --------------------------
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "Weights" / "best.pt"
output_dir = base_dir / "Media" / "Image_Result"
output_dir.mkdir(parents=True, exist_ok=True)

# Load model
yolo_model = YOLO(model_path)

# Class labels
class_labels = [
    'Front-Windscreen-Damage', 'Headlight-Damage', 'Major-Rear-Bumper-Dent',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'doorouter-dent',
    'fender-dent', 'front-bumper-dent', 'medium-Bodypanel-Dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read image from file stream
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # Run YOLO
    results = yolo_model(img)
    predictions = []

    # Annotate detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = round(float(box.conf[0]), 2)
            cls_id = int(box.cls[0])

            if conf > 0.3:
                label = class_labels[cls_id]
                predictions.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{label} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

    # Save annotated image
    save_path = output_dir / f"{Path(filename).stem}_predicted.jpg"
    cv2.imwrite(str(save_path), img)

    return jsonify({
        "filename": filename,
        "result_image_path": str(save_path.relative_to(base_dir)),
        "detections": predictions
    })


@app.route('/')
def home():
    return 'YOLO Damage Detection API is running!'


if __name__ == '__main__':
    app.run(debug=True)
