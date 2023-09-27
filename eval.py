import numpy as np
import cv2
import os
import tensorflow as tf
from keras.models import load_model

# Define the video-to-frames function
def video_to_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128))
        frames.append(frame)
    cap.release()
    return np.stack(frames)

# Load the trained model
model_path = os.path.join(r"C:\Users\s.apraez\Emprendimiento", 'MEDTRACK.h5')
model = load_model(model_path)

# Process the video
video_path = r"C:\Users\s.apraez\Emprendimiento\Video_classification\train\1.0\(1.0)(7).mp4"
num_frames = 5  # Adjust if needed
video_data = video_to_frames(video_path, num_frames)
video_data = np.expand_dims(video_data, axis=0)  # Add batch dimension

# Make prediction
predictions = model.predict(video_data)
predicted_class = np.argmax(predictions[0])
print(predictions[0])
# Mapping of class indices to frequencies
frequencies = [1.0, 1.0526, 1.1053, 1.1579, 1.2105, 1.2632, 1.3158, 1.3684, 1.4211, 1.4737, 1.5263, 1.5789, 1.6316, 1.6842, 1.7368, 1.7895, 1.8421, 1.8947, 1.9474, 2.0]  # Ensure this list is accurate and complete
predicted_frequency = frequencies[predicted_class]

print(f"The video belongs to the frequency: {predicted_frequency}")