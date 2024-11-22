import cv2
import mediapipe as mp
import numpy as np
import os
import json
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_pose_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pose_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]
            pose_landmarks.append(landmarks)
    
    cap.release()
    return pose_landmarks

data_dir = r'C:\Users\Shanta Das\Desktop\model training\data'  # Update this path to your actual data directory
labels = ['violent', 'non-violent']
data = []

for label in labels:
    path = os.path.join(data_dir, label)
    if not os.path.exists(path):
        print(f"Directory {path} does not exist.")
        continue
    for video_name in os.listdir(path):
        video_path = os.path.join(path, video_name)
        print(f"Processing video: {video_path}")
        try:
            landmarks_list = extract_pose_landmarks_from_video(video_path)
            for landmarks in landmarks_list:
                data.append([json.dumps(landmarks), label])
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

# Save data to a CSV file
df = pd.DataFrame(data, columns=['landmarks', 'label'])
csv_path = os.path.join(data_dir, 'pose_data.csv')
df.to_csv(csv_path, index=False)

print(f"Data processing complete. Pose landmarks saved to '{csv_path}'.")
