import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import cv2

# Function to update the frame
def update_frame(index):
    ax.clear()
    ax.imshow(frames[index])
    fig.canvas.draw()

# Function to handle key press events
def on_key(event):
    global frame_index
    if event.key == 'left':  # Left arrow key
        frame_index = max(0, frame_index - 1)
    elif event.key == 'right':  # Right arrow key
        frame_index = min(len(frames) - 1, frame_index + 1)
    update_frame(frame_index)

# Load video frames using OpenCV
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)
frames = []

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frames.append(frame)

cap.release()

if not frames:
    raise ValueError("No frames were loaded from the video.")

frame_index = 0

# Set up the plot
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_key)

# Display the first frame
update_frame(frame_index)

# Extract audio from the video
audio = AudioSegment.from_file(video_path)

# Convert audio to numpy array
audio_samples = np.array(audio.get_array_of_samples())
sample_rate = audio.frame_rate

# Analyze the audio to detect spikes
# Here we use a simple threshold method to detect spikes
threshold = np.max(audio_samples) * 0.6
spikes = np.where(audio_samples > threshold)[0]

# Log the detected spikes with video time and frame
for spike in spikes:
    time_in_seconds = spike / sample_rate
    corresponding_frame = int(time_in_seconds * fps)
    print(f"Detected spike at index {spike}, time {time_in_seconds:.2f}s, frame {corresponding_frame}")

# Plot the audio signal and detected spikes
plt.figure(figsize=(12, 6))
plt.plot(audio_samples, label='Audio Signal')
plt.scatter(spikes, audio_samples[spikes], color='red', label='Detected Spikes')
plt.title('Audio Signal with Detected Spikes')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Show the plot
plt.show()