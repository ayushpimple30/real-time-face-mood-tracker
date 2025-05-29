import cv2
from deepface import DeepFace
from datetime import datetime
import csv
import os

# Use text-based emojis (since OpenCV can't show Unicode emojis)
emoji_map = {
    'happy': ':)',
    'sad': ':(',
    'angry': '>:(',
    'surprise': ':O',
    'neutral': ':|',
    'fear': ':-S',
    'disgust': 'XP'
}

# Create log file if not exists
log_file = 'mood_log.csv'
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Mood'])

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 's' to save screenshot, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze emotions
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Handle single or multiple faces
        if isinstance(results, list):
            faces = results
        else:
            faces = [results]

        for face in faces:
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w']
            h = face['region']['h']
            emotion = face['dominant_emotion']
            emoji = emoji_map.get(emotion, '')

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{emotion} {emoji}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Log to CSV
            with open(log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), emotion])

    except Exception as e:
        print("[WARNING] No face detected or error:", str(e))

    # Display the frame
    cv2.imshow("Live Mood Tracker", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save screenshot
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Screenshot saved as {filename}")

# Release everything
cap.release()
cv2.destroyAllWindows()
