import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import time

st.set_page_config(layout="centered")
st.title("🎭 Real-Time Face Mood Detector")

start_cam = st.checkbox("Start Webcam")

frame_window = st.empty()
status_text = st.empty()

if start_cam:
    cap = cv2.VideoCapture(0)

    while start_cam:
        ret, frame = cap.read()
        if not ret:
            status_text.error("Failed to grab frame")
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            label = f"Mood: {emotion}"
            color = (0, 255, 0)
        except:
            label = "No face detected"
            color = (0, 0, 255)

        # Draw label on frame
        cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)
        time.sleep(0.1)

    cap.release()
    st.write("✅ Camera stopped.")
else:
    st.write("👈 Check the box to start webcam.")
