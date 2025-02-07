import os
import cv2
import time
import pickle
import tempfile
import numpy as np
import face_recognition
from gtts import gTTS
import pygame
from datetime import datetime

# Initialize pygame mixer for audio feedback
pygame.mixer.init()

dataset_dir = r"C:\Users\Lenovo\Downloads\mysigth\face_recognition\dataset"
model_path = "face_recognition_model.pkl"

def speak(text):
    print(f"Speaking: {text}")
    tts = gTTS(text=text, lang='en')
    temp_file = os.path.join(tempfile.gettempdir(), "feedback.mp3")
    tts.save(temp_file)
    
    sound = pygame.mixer.Sound(temp_file)
    sound.play()
    while pygame.mixer.get_busy():  # Wait for the audio to finish
        time.sleep(0.1)
    
    os.remove(temp_file)  # Clean up temp file

def capture_photos(name):
    folder = os.path.join(dataset_dir, name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    cap = cv2.VideoCapture(1)  # Try USB webcam
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Fallback to default
    
    if not cap.isOpened():
        speak("Error: Could not open webcam.")
        return
    
    time.sleep(2)
    photo_count = 0
    speak(f"Taking photos for {name}. Press SPACE to capture, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Error: Failed to capture frame.")
            break

        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Space key
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            speak(f"Photo {photo_count} saved.")

        elif key == ord('q'):  # Q key
            break

    cap.release()
    cv2.destroyAllWindows()
    speak(f"Photo capture completed. {photo_count} photos saved for {name}.")

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

def recognize_faces():
    if not os.path.exists(model_path):
        speak("No trained model found. Please train the model first.")
        return
    
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    speak("Starting facial recognition.")
    cap = cv2.VideoCapture(1)  # Try USB webcam
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Fallback to default
    
    if not cap.isOpened():
        speak("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Error: Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Unknown"
            
            if True in matches:
                matched_indexes = [i for i, match in enumerate(matches) if match]
                name_counts = {}
                
                for i in matched_indexes:
                    name = data["names"][i]
                    name_counts[name] = name_counts.get(name, 0) + 1
                
                name = max(name_counts, key=name_counts.get)
                speak(f"Face recognized: {name}")

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Facial Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    speak("Facial recognition session ended.")

if __name__ == "__main__":
    while True:
        print("Select an option:")
        print("1. Capture Photos")
        print("2. Train Model")
        print("3. Recognize Faces")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            name = input("Enter the person's name: ").strip()
            capture_photos(name)
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            speak("Exiting application.")
            break
        else:
            speak("Invalid choice. Please try again.")
