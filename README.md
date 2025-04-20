# 🚗 AI Car Damage Detection

This project leverages computer vision and deep learning to automatically detect and assess damage on cars from images. It is designed to assist insurance companies, car rental services, and automobile manufacturers in automating the inspection process.

## 🔍 Features

- Detects visible car damage ( dents, broken parts).
- Localizes damaged areas using bounding boxes and masks.
- Classifies damage types (e.g. dent).
- Generates confidence scores and visual reports.
- Modular architecture for easy model updates or integration with APIs.

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Frameworks**: PyTorch / TensorFlow, OpenCV, FlasktAPI 
- **Deep Learning Models**: YOLOv8 
- **Deployment**: Docker, Flask,gradio
- **Tools**: NumPy, Pandas, Matplotlib, Albumentations

## 📁 Project Structure

AI_car_damage_detection/
│
├── Media/
│   ├── Image_result/             # Output images with predictions and bounding boxes
│   ├── testing_media_file/       # Input images and videos for testing
│   └── Video_result/             # Output videos with predictions
│
├── api_ui_screenshot/            # Screenshots of UI and API responses
│
├── car_damage_image.py           # Script to run prediction on image input
├── car_damage_image_ui.py        # UI for uploading image and getting prediction
├── car_damage_video.py           # Script to run prediction on video input
├── car_damage_video_ui.py        # UI for uploading video and getting prediction
│
├── image_app.py                  # Flask API for image prediction
├── api_request.py                # Script to send API requests (can use Postman too)
│
├── requirements.txt              # Dependencies for the project
└── README.md                     # Project documentation

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/iamabhishekPande/AI_Car_Damage_Detection.git
cd AI_car_damage_detection
2. Install dependencies
    pip install -r requirements.txt
🚀 How to Run
🔹 A. Image Prediction (Script-based)

    1. Add your image to the Media/testing_media_file/ folder.
    2. Open car_damage_image.py and set the image path.
    3. Run the script using Python: python car_damage_image.py
    4. The output image with predictions will be saved in Media/Image_result/

B. Image Prediction via UI
    1. Run the Flask API using: python image_app.py
    2. Upload your image through the interface.
    3. Prediction results will be displayed directly in the interface.
 C. Video Prediction (Script-based)
    1. Add your video to the Media/testing_media_file/ folder.
    2. Open car_damage_video.py and set the video path.
    3. Run the script using Python: python car_damage_video.py
    4. The output video with predictions will be saved in Media/Video_result/
D. Video Prediction via UI
    1. Run the Flask API using: python car_damage_video_ui.py
    2. Upload your video through the interface.
    3. Prediction results will be displayed directly in the interface.

E. Using Flask API
    1. start the Flask API using: python image_app.py
    2. Send a POST request with an image file using api_request.py or Postman
    3. The API will return the prediction results in json.
    4. Refer to api_ui_screenshot/ for sample API response screenshots.

💬 Notes
        Make sure the YOLO model weights are correctly loaded in your scripts.
        Ensure that the required image/video paths are set before running the scripts.
        Results are saved automatically in their respective result folders for review.
📸 Screenshots
        All screenshots of the UI and API responses are located in the api_ui_screenshot/ folder for reference.
📌 Dependencies
        Install dependencies using:pip install -r requirements.txt



    

    


