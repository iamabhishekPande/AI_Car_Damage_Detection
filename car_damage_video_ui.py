import cv2
import time
import cvzone
import gradio as gr
import tempfile
from ultralytics import YOLO
from pathlib import Path

# --------------------------
# Configuration
# --------------------------
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "Weights" / "best.pt"
resize_width, resize_height = 640, 360
skip_every = 5
confidence_threshold = 0.3

# --------------------------
# Load YOLO Model
# --------------------------
yolo_model = YOLO(str(model_path))

# --------------------------
# Class Labels
# --------------------------
class_labels = [
    'Front-Windscreen-Damage', 'Headlight-Damage', 'Major-Rear-Bumper-Dent',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'doorouter-dent',
    'fender-dent', 'front-bumper-dent', 'medium-Bodypanel-Dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

# --------------------------
# Process Function
# --------------------------
def detect_damage(video_file, confidence=0.3, skip=5):
    global confidence_threshold, skip_every
    confidence_threshold = confidence
    skip_every = skip

    # Use a temporary file for output
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return "Failed to open video", None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, 20.0, (resize_width, resize_height))
    
    frame_count = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % skip_every != 0:
            continue

        img = cv2.resize(img, (resize_width, resize_height))

        results = yolo_model(img, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                if conf > confidence_threshold:
                    label = class_labels[cls_id]
                    cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                    cvzone.putTextRect(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                                       scale=0.7, thickness=1, colorR=(255, 0, 0))

        out.write(img)

    cap.release()
    out.release()

    return "Processing complete!", temp_output.name

# --------------------------
# Gradio UI
# --------------------------
demo = gr.Interface(
    fn=detect_damage,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Confidence Threshold"),
        gr.Slider(1, 10, value=5, step=1, label="Skip Every N Frames")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Video(label="Processed Output")
    ],
    title="Car Damage Detection with YOLOv8",
    description="Upload a video to detect car part damages using a trained YOLOv8 model."
)

if __name__ == "__main__":
    demo.launch()
