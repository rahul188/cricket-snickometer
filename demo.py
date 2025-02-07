import cv2
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np

# Open the video file
cap = cv2.VideoCapture('input.mp4')

# Get frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the number of frames to skip (first 2 seconds)
frames_to_skip = 0

# Read all frames into a list, skipping the first 2 seconds
frames = []
frame_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_index >= frames_to_skip:
        frames.append(frame)
    frame_index += 1

# Initialize frame index
frame_index = 0

def update_frame(index):
    frame = frames[index]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Calculate current time
    current_time = (index + frames_to_skip) / fps
    
    # Overlay frame number and current time on the frame with increased font size
    font_scale = 2  # Increase this value to make the font larger
    font_color = (0,0,255)
    cv2.putText(frame_rgb, f'Frame: {index}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, font_scale,font_color, 2, cv2.LINE_AA)
    cv2.putText(frame_rgb, f'Time: {current_time:.2f}s', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, font_scale,font_color, 2, cv2.LINE_AA)
    
    ax.clear()
    ax.imshow(frame_rgb)
    plt.draw()

def on_key(event):
    global frame_index
    if event.key == 'left':  # Left arrow key
        frame_index = max(0, frame_index - 1)
    elif event.key == 'right':  # Right arrow key
        frame_index = min(len(frames) - 1, frame_index + 1)
    update_frame(frame_index)

# Set up the plot
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_key)

# Display the first frame
update_frame(frame_index)

# Extract audio from the video
audio = AudioSegment.from_file('input.mp4')

# Convert audio to numpy array
audio_samples = np.array(audio.get_array_of_samples())

# Analyze the audio to detect spikes
# Here we use a simple threshold method to detect spikes
threshold = np.max(audio_samples) * 0.6
spikes = np.where(audio_samples > threshold)[0]

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

# Release the video capture object
cap.release()