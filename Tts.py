import os
import time
import tempfile
from gtts import gTTS
import pygame

# Initialize pygame mixer
pygame.mixer.init()

def speak(text):
    """Convert text to speech and play the audio."""
    print(f"Speaking: {text}")
    tts = gTTS(text=text, lang='en')
    temp_file = os.path.join(tempfile.gettempdir(), "feedback.mp3")
    tts.save(temp_file)
    
    # Load and play the audio using pygame
    sound = pygame.mixer.Sound(temp_file)
    sound.play()
    while pygame.mixer.get_busy():  # Wait for the audio to finish playing
        time.sleep(0.1)
    
    os.remove(temp_file)  # Delete the file after playing

if __name__ == "__main__":
    text = input("Enter text to speak: ")
    speak(text)
