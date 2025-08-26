import cv2
import os
import numpy as np

# Function to prepare training data
def prepare_training_data(data_folder_path):
    labels = []
    faces = []
    names = {}

    for label, folder_name in enumerate(os.listdir(data_folder_path)):
        folder_path = os.path.join(data_folder_path, folder_name)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray_image is not None:
                faces.append(gray_image)
                labels.append(label)
        names[label] = folder_name

    return np.array(faces), np.array(labels), names

# Function to detect and recognize faces
def recognize_faces(faces, labels, names, data_folder_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video. Exiting.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces_detected:
            face = gray_frame[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face)
            name = names.get(label, "Unknown")

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display name and confidence
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to folder containing criminal images
    data_folder_path = r"D:\SIH-2024-Project---Women-Safety-USing-CCTV--main (2)\flask\crime_data"

    # Prepare training data
    faces, labels, names = prepare_training_data(data_folder_path)

    # Recognize faces in live video
    recognize_faces(faces, labels, names, data_folder_path)
