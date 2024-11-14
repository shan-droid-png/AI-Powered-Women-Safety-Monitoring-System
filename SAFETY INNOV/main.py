import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array

# Function to load images and labels from a folder
def load_data(dataset_path, img_size=(224, 224), frames_per_video=30):
    images = []
    labels = []
    
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            # Assuming each video has a folder with the same name as the label (violent/non-violent)
            video_path = os.path.join(subdir, file)
            
            # Load video frames
            video_frames = extract_frames_from_video(video_path, frames_per_video, img_size)
            if len(video_frames) == frames_per_video:  # Ensure we have the desired number of frames
                images.append(video_frames)
                label = 0 if 'non-violent' in subdir else 1  # Label: 0 = non-violent, 1 = violent
                labels.append(label)
                
    return np.array(images), np.array(labels)

# Function to extract frames from video
def extract_frames_from_video(video_path, frames_per_video, img_size):
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Resize and normalize the frame
        frame_resized = cv2.resize(frame, img_size)
        frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
        frames.append(frame_normalized)
        if len(frames) == frames_per_video:
            break
    video.release()
    return frames

# Model for violence detection using 3D CNN
def build_violence_detection_model(input_shape=(224, 224, 30, 3)):
    model = models.Sequential()
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.Conv3D(128, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training the model
def train_model(dataset_path, img_size=(224, 224), frames_per_video=30, epochs=10, batch_size=32):
    images, labels = load_data(dataset_path, img_size, frames_per_video)
    
    model = build_violence_detection_model(input_shape=(img_size[0], img_size[1], frames_per_video, 3))
    model.summary()
    
    model.fit(images, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    # Save the trained model
    model.save('violence_detection_model.h5')
    return model

# Real-time camera detection
def real_time_violence_detection(model):
    cap = cv2.VideoCapture(0)  # Open the webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame (resize and normalize)
        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized / 255.0
        frame_input = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
        
        # Predict violence
        prediction = model.predict(frame_input)
        label = "Violent" if prediction[0] > 0.5 else "Non-Violent"
        color = (0, 0, 255) if prediction[0] > 0.5 else (0, 255, 0)
        
        # Display the prediction
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow("Violence Detection", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    dataset_path = 'C:\\Users\\dsneh\\OneDrive\\Desktop\\SIH 2024\\Action-Recognition-master'  # Replace with the actual dataset path
    
    # Uncomment this line to train the model (only once)
    # model = train_model(dataset_path)
    
    # Load the trained model for real-time detection
    model = tf.keras.models.load_model('violence_detection_model.h5')
    
    # Run real-time violence detection
    real_time_violence_detection(model)

if __name__ == "__main__":
    main()
