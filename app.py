import cv2
import numpy as np
import pyaudio
from scipy.signal import find_peaks

# Video capture setup
cap = cv2.VideoCapture('input.mp4')  # Replace with the path to your video file

# Audio capture setup
p = pyaudio.PyAudio()

# List available audio devices
print("Available audio devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"{i}: {info['name']}")

# Select the appropriate input device index
input_device_index = int(input("Enter the input device index: "))

# Check if the selected device supports the required number of channels
device_info = p.get_device_info_by_index(input _device_index)
if device_info['maxInputChannels'] < 1:
    print(f"Selected device does not support mono audio input.")
    p.terminate()
    exit(1)

try:
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024, input_device_index=input_device_index)
except OSError as e:
    print(f"Could not open audio stream: {e}")
    p.terminate()
    exit(1)

threshold = 3000  # Threshold for detecting snick in audio

# Video writer setup to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

print("Starting video processing...")

try:
    while cap.isOpened():
        ret, frame = cap.read()  # Capture video frame by frame
        if not ret:
            break

        # Capture audio data
        audio_data = stream.read(1024)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Detect snick by checking for peaks in sound intensity
        peaks, _ = find_peaks(audio_array, height=threshold)
        
        # Display detected snick on the video
        if len(peaks) > 0:
            cv2.putText(frame, "Snick Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display video frame with snick detection
        cv2.imshow("Video Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Video processing ended.")
finally:
    # Release resources
    cap.release()
    out.release()
    stream.stop_stream()
    stream.close()
    p.terminate()
    cv2.destroyAllWindows()
