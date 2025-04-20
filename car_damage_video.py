import cv2
import time
import cvzone
from ultralytics import YOLO
from pathlib import Path

# --------------------------
# Configuration
# --------------------------
base_dir = Path(__file__).resolve().parent  # Directory where script is located

# Paths
model_path = base_dir / "Weights" / "best.pt"
video_path = base_dir / "Media" / "testing_media_file" / "damageCar.mp4"
input_filename = Path(video_path).stem  # e.g., "damageCar"
output_filename = f"{input_filename}_prediction.mp4"
output_dir=base_dir/"Media"/"Video_result"
output_path = output_dir / output_filename

#output_path = base_dir / "Media" / "output_prediction.mp4"  # Output video path

# Settings
resize_width, resize_height = 640, 360
skip_every = 5
confidence_threshold = 0.3
show_fps = True

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
# Load Video
# --------------------------
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

# Setup Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (resize_width, resize_height))

frame_count = 0

# --------------------------
# Process Video Frames
# --------------------------
while True:
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % skip_every != 0:
        continue

    img = cv2.resize(img, (resize_width, resize_height))
    start_time = time.time()

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
                cvzone.putTextRect(img, f'{label} {conf:.2f}', (x1, y1 - 10), scale=0.7, thickness=1, colorR=(255, 0, 0))

    # Show FPS
    # if show_fps:
    #     fps = 1 / (time.time() - start_time + 1e-5)
    #     cv2.putText(img, f'FPS: {fps:.1f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show and Save Frame
    cv2.imshow("Fast YOLO Video Detection", img)
    out.write(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------
# Cleanup
# --------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
