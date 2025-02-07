# Face Recognition with Audio Feedback

This project is a facial recognition system that provides real-time audio feedback using text-to-speech (TTS). It captures images, trains a model, and recognizes faces through a webcam.

## Features
- **Face recognition** using OpenCV and face_recognition library
- **Audio feedback** using gTTS and pygame
- **Live webcam detection** for real-time recognition
- **Dataset management** with automatic folder creation

---

## Installation

### Prerequisites
Ensure you have **Python 3.x** installed. Then, install the required dependencies:
```bash
pip install opencv-python numpy face-recognition gtts pygame
```

### Setup Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/zeyadabdelwahab17/Face-recognition-with-audio-feedback.git
   cd Face-recognition-with-audio-feedback
   ```

2. **Create a Dataset Folder**
   Inside the project directory, create a folder named `dataset`:
   ```bash
   mkdir dataset
   ```
   Copy the full path of this folder and update it in `image_capture.py` and `model_training.py`:
   ```python
   dataset_dir = "C:/path/to/your/dataset"
   ```

3. **Capture Images**
   Run the script to capture images for training:
   ```bash
   python image_capture.py
   ```
   Enter a name when prompted, then press `SPACE` to take pictures and `Q` to quit.

4. **Train the Model**
   Once images are collected, train the model:
   ```bash
   python model_training.py
   ```
   This will generate a `face_recognition_model.pkl` file.

5. **Run Face Recognition**
   To start real-time face recognition:
   ```bash
   python face_recognition.py
   ```
   The webcam will open, and detected faces will be announced via audio feedback.

---

## Usage Notes
- The **dataset** folder is ignored in `.gitignore` and won’t be pushed to GitHub.
- The project is designed to work with a **webcam** (USB or built-in).
- The `tts.py` script handles text-to-speech conversion.

## Future Improvements
- Improve accuracy with deep learning models.
- Add a mobile app for remote face recognition.
- Support multiple languages for TTS.

Feel free to contribute or modify the project!



