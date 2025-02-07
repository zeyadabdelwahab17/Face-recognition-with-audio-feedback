import cv2
import face_recognition
import pickle
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# Load the trained face recognition model
model_path = "face_recognition_model.pkl"

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("feedback.mp3")
    sound = AudioSegment.from_mp3("feedback.mp3")
    play(sound)
    os.remove("feedback.mp3")

# Load the trained model
with open(model_path, "rb") as f:
    data = pickle.load(f)

def recognize_faces():
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
    recognize_faces()
