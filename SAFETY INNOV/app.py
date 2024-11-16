from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
import time
import json

app = Flask(__name__)

# Load the trained violence detection model
model = tf.keras.models.load_model('violence_detection_model.h5')

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained models for gender detection
gender_net = cv2.dnn.readNetFromCaffe(r'C:\Users\Shanta Das\Desktop\html\models\deploy_gender.prototxt', r'C:\Users\Shanta Das\Desktop\html\models\gender_net.caffemodel')
gender_list = ['Male', 'Female']

# Alarm system flag
alarm_triggered = False

def extract_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
        return np.array(landmarks).flatten()
    return None

def gen_frames():
    vs = cv2.VideoCapture(0)
    while vs.isOpened():
        ret, frame = vs.read()
        if not ret:
            break

        # Extract pose landmarks
        landmarks = extract_pose_landmarks(frame)
        global alarm_triggered
        if landmarks is not None:
            landmarks = np.expand_dims(landmarks, axis=0)
            prediction = model.predict(landmarks)[0][0]
            label = 'Violent' if prediction > 0.5 else 'Non-Violent'
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if label == 'Violent':
                alarm_triggered = True  # Trigger alarm if a violent act is detected

        # Haar Cascade for face detection and gender detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        detected_genders = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            detected_genders.append(gender)
            label = f"{gender}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Check for alarm condition (female alone)
        if detected_genders == ['Female']:
            alarm_triggered = True

        # Display frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    vs.release()

def alarm_system():
    global alarm_triggered
    while True:
        if alarm_triggered:
            print("Alarm! Person detected alone or violent act detected.")
            alarm_triggered = False
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=alarm_system).start()  # Start the alarm system in a separate thread
    app.run(debug=True)
