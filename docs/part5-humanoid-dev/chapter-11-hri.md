---
title: "Chapter 11: Human-Robot Interaction for Humanoids"
sidebar_position: 11
---

# Chapter 11: Human-Robot Interaction for Humanoids

## Introduction to HRI for Humanoid Robots

Human-Robot Interaction (HRI) is a critical aspect of humanoid robotics that focuses on enabling natural and intuitive communication between humans and robots. This chapter explores the key components, challenges, and implementation strategies for effective HRI in humanoid robots.

## 11.1 Multimodal Interaction

### 11.1.1 Speech Recognition and Synthesis

```python
import speech_recognition as sr
import pyttsx3
import numpy as np

class SpeechInterface:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        
    def listen(self, timeout=5, phrase_time_limit=5):
        """Listen for voice input and convert to text"""
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                text = self.recognizer.recognize_google(audio)
                return text.lower()
            except (sr.UnknownValueError, sr.WaitTimeoutError):
                return None
    
    def speak(self, text, voice_index=0):
        """Convert text to speech"""
        self.engine.setProperty('voice', self.voices[voice_index].id)
        self.engine.say(text)
        self.engine.runAndWait()
    
    def set_voice(self, gender='female'):
        """Set voice gender"""
        if gender.lower() == 'male':
            self.engine.setProperty('voice', self.voices[0].id)
        else:
            self.engine.setProperty('voice', self.voices[1].id)
    
    def adjust_speech_rate(self, rate=200):
        """Adjust speech rate (words per minute)"""
        self.engine.setProperty('rate', rate)

# Example usage
if __name__ == "__main__":
    speech = SpeechInterface()
    speech.speak("Hello! How can I assist you today?")
    command = speech.listen()
    if command:
        print(f"You said: {command}")
        speech.speak(f"I heard you say: {command}")