
import cv2
import math
import os
import cvzone
from ultralytics import YOLO
from pathlib import Path

# --------------------------
# Load YOLO model
# --------------------------
base_dir = Path(__file__).resolve().parent  # Directory where script is located
model_path = base_dir / "Weights" / "best.pt"
yolo_model = YOLO(model_path)

# --------------------------
# Class names (in order)
# --------------------------
class_labels = [
    'Front-Windscreen-Damage', 'Headlight-Damage', 'Major-Rear-Bumper-Dent',
    'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage',
    'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'doorouter-dent',
    'fender-dent', 'front-bumper-dent', 'medium-Bodypanel-Dent', 'pillar-dent',
    'quaterpanel-dent', 'rear-bumper-dent', 'roof-dent'
]

# --------------------------
# Load Image
# --------------------------
image_path = base_dir / "Media" / "testing_media_file" / "2.jpg"
img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# --------------------------
# Run YOLOv8 detection
# --------------------------
results = yolo_model(img)

# --------------------------
# Annotate detections
# --------------------------
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

# --------------------------
# Save Annotated Image
# --------------------------
# Create output directory if it doesn't exist
output_dir = base_dir / "Media" / "Image_Result"
output_dir.mkdir(parents=True, exist_ok=True)

# Save with original filename + _predicted suffix
output_filename = image_path.stem + "_predicted" + image_path.suffix
output_path = output_dir / output_filename

cv2.imwrite(str(output_path), img)
print(f"Annotated image saved to: {output_path}")

# --------------------------
# Show Result
# --------------------------
cv2.imshow("Detected Image", img)
print("Press 'q' to close the window.")

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
