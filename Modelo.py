import numpy as np
import cv2
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout

# Paths
path = r"C:\Users\s.apraez\Emprendimiento\Video_classification"
train_path = os.path.join(path, "train")
val_path = os.path.join(path, "test")

# Assuming videos are in .avi format for this example
def video_to_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 128))  # Reduced resolution
        frames.append(frame)
    cap.release()
    return np.stack(frames)

def VideoDataGenerator(dir_path, batch_size=10, num_frames=5):
    while True:
        video_folders = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        batch_videos, batch_labels = [], []
       
        for i in range(batch_size):
            random_folder = np.random.choice(video_folders)
            videos_in_folder = [f for f in os.listdir(os.path.join(dir_path, random_folder)) if f.endswith('.mp4')]
            random_video = np.random.choice(videos_in_folder)
           
            video_path = os.path.join(dir_path, random_folder, random_video)
            frames = video_to_frames(video_path, num_frames)
            label = video_folders.index(random_folder)
           
            batch_videos.append(frames)
            batch_labels.append(label)
       
        yield np.stack(batch_videos), tf.keras.utils.to_categorical(batch_labels, num_classes=len(video_folders))


train_info = VideoDataGenerator(train_path)
val_data = VideoDataGenerator(val_path)

# Model definition
model = Sequential()

# First Conv3D layer
model.add(Conv3D(filters=8, activation="relu", kernel_size=(3, 3, 3), input_shape=(5, 128, 128, 3)))  # Reduced filters and adjusted input shape
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

# Second Conv3D layer
model.add(Conv3D(filters=32, activation="relu", kernel_size=(3, 3, 3)))  # Reduced filters
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(Dropout(0.5))  # Added dropout for regularization

# Flatten and Fully Connected layers
model.add(Flatten())
model.add(Dense(64, activation="relu"))  # Reduced units
model.add(Dropout(0.5))  # Added dropout for regularization
model.add(Dense(20, activation="softmax"))

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
model.fit(train_info, epochs=10, validation_data=val_data, steps_per_epoch=100, validation_steps=50)

model.save(os.path.join(r"C:\Users\s.apraez\Emprendimiento", 'MEDTRACK.h5'))