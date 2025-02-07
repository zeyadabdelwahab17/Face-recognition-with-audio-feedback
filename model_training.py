import os
import cv2
import numpy as np
import face_recognition
import pickle
from gtts import gTTS
import time
import pygame
import tempfile

# Initialize pygame mixer
pygame.mixer.init()

# Define dataset and model paths
dataset_dir = r"C:\Users\Lenovo\Downloads\mysigth\face_recognition\dataset"
model_path = "face_recognition_model.pkl"

def speak(text):
    print(f"Current working directory: {os.getcwd()}")
    tts = gTTS(text=text, lang='en')
    temp_file = os.path.join(tempfile.gettempdir(), "feedback.mp3")
    print(f"Saving file to: {temp_file}")
    tts.save(temp_file)
    
    # Load and play the audio using pygame.mixer.Sound
    sound = pygame.mixer.Sound(temp_file)
    sound.play()
    while pygame.mixer.get_busy():  # Wait for the audio to finish playing
        time.sleep(0.1)
    
    os.remove(temp_file)  # Delete the file after playing

def train_model():
    encodings = []
    names = []
    
    speak("Starting face recognition model training.")
    
    for person_name in os.listdir(dataset_dir):
        person_folder = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            for encoding in face_encodings:
                encodings.append(encoding)
                names.append(person_name)
    
    data = {"encodings": encodings, "names": names}
    with open(model_path, "wb") as f:
        pickle.dump(data, f)
    
    speak("Model training completed successfully.")

if __name__ == "__main__":
    train_model()