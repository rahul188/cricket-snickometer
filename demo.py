import cv2
import matplotlib.pyplot as plt

# Open the video file
cap = cv2.VideoCapture('input.mp4')

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Read all frames into a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # Initialize frame index
    frame_index = 0

    def update_frame(index):
        frame = frames[index]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate current time
        current_time = index / fps
        
        # Overlay frame number and current time on the frame
        cv2.putText(frame_rgb, f'Frame: {index}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_rgb, f'Time: {current_time:.2f}s', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
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

    # Connect the key press event to the handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Display the first frame
    update_frame(frame_index)
    
    # Wait for key press to update frames
    while True:
        if plt.waitforbuttonpress():
            continue

    # Release the video capture object
    cap.release()
    # Close all matplotlib windows