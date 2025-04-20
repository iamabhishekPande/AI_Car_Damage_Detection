import cv2
import cvzone
import gradio as gr
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# --------------------------
# Load YOLO model and class names
# --------------------------
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "Weights" / "best.pt"
yolo_model = YOLO(model_path)

class_labels = [
    'Front-Windscreen-Damage', 'Headlight-Damage', 'Major-Rear-Bumper-Dent',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'doorouter-dent',
    'fender-dent', 'front-bumper-dent', 'medium-Bodypanel-Dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

# --------------------------
# Detection Function
# --------------------------
def detect_damage(image: Image.Image):
    # Convert to OpenCV format (numpy array)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = yolo_model(img)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = round(float(box.conf[0]), 2)
            cls_id = int(box.cls[0])

            if conf > 0.3:
                label = class_labels[cls_id]
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{label} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0))

    # Convert back to RGB for Gradio display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# --------------------------
# Gradio UI
# --------------------------
demo = gr.Interface(
    fn=detect_damage,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil", label="Detected Image"),
    title="🚗 Vehicle Damage Detection",
    description="Upload an image of a vehicle to detect damage using YOLOv8."
)

if __name__ == "__main__":
    demo.launch()
