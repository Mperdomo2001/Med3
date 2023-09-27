import cv2

# Load the video
video_path = r"C:\Users\s.apraez\Emprendimiento\Video_classification\test\(1.1053)(2).mp4"  # replace with the path to one of your videos
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Print the details
print(f"Number of Frames: {frame_count}")
print(f"Width: {width}")
print(f"Height: {height}")
print(f"FPS: {fps}")

cap.release()
