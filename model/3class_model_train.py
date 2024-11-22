import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import json

# Load data
df = pd.read_csv(r'C:\Users\Shanta Das\Desktop\model training\data\pose_data.csv')  # Update this path if needed

# Parse landmarks safely
X = np.array([json.loads(landmarks) for landmarks in df['landmarks']])
X = X.reshape(X.shape[0], -1)  # Flatten the list of lists
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert labels to one-hot encoding for 3-class classification
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # Updated for 3-class classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save the model
model.save('violence_detection_model.h5')

print("Model training complete and saved as 'violence_detection_model.h5'.")
