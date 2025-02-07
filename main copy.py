import cv2
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import wave
import contextlib
import os

# Load video file
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)

# Extract audio from video
audio_path = 'audio.wav'
command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y"
os.system(command)

# Load audio file
audio = AudioSegment.from_wav(audio_path)

# Get frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get duration of the audio
with contextlib.closing(wave.open(audio_path, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

# Calculate the duration of each frame in milliseconds
frame_duration_ms = (duration / total_frames) * 1000

# Initialize previous frame
prev_frame = None

# Process each frame
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for difference calculation
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # Compute the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        
        # Display the difference
        cv2.imshow('Frame Difference', frame_diff)
        
        # Print the sum of absolute differences as a simple metric
        print(f'Frame {frame_number} Difference: {np.sum(frame_diff)}')

    # Update previous frame
    prev_frame = gray_frame

    # Extract the corresponding audio segment
    start_time = frame_number * frame_duration_ms
    end_time = start_time + frame_duration_ms
    audio_segment = audio[start_time:end_time]

    # Convert audio segment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())

    # Plot the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(samples)
    plt.title(f'Frame {frame_number}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

    # Display the video frame
    cv2.imshow('Video Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()