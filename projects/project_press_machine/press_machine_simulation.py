import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_press_frame(frame_size, rect_width, rect_height, rect_y_position):
    """Create a single frame with a rectangle at the specified position"""
    frame = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255
    
    # Calculate rectangle position (center it horizontally)
    rect_x = (frame_size[1] - rect_width) // 2
    
    # Draw the rectangle
    cv2.rectangle(frame, (rect_x, rect_y_position), 
                 (rect_x + rect_width, rect_y_position + rect_height), 
                 (0, 0, 0), -1)  # Black filled rectangle
    
    return frame

def detect_press(frame):
    """Detect the press in the frame and calculate its area"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours found, process the largest one
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate area
        area = cv2.contourArea(largest_contour)
        
        # Draw contour
        frame_with_contour = frame.copy()
        cv2.drawContours(frame_with_contour, [largest_contour], -1, (0, 0, 255), 2)
        
        # Display area text
        cv2.putText(frame_with_contour, f"Area: {int(area)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame_with_contour, area
    
    return frame, 0

# Parameters
frame_size = (400, 600)  # height, width
rect_width = 200
min_rect_height = 50
max_rect_height = 200
area_threshold = 15000  # Threshold to determine press state
total_frames = 120

# Initialize variables to track press state and count
press_count = 0
prev_press_state = "OPEN"  # Start with press open

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('press_simulation.avi', fourcc, 30.0, (frame_size[1], frame_size[0]))

# Generate frames
all_frames = []
areas = []
press_states = []

for i in range(total_frames):
    # Simulate press movement (down then up)
    if i < total_frames // 2:
        # Press going down (rectangle gets bigger)
        completion = i / (total_frames // 2)
        rect_height = int(min_rect_height + completion * (max_rect_height - min_rect_height))
        rect_y = 100  # Fixed top position
    else:
        # Press going up (rectangle gets smaller)
        completion = (i - total_frames // 2) / (total_frames // 2)
        rect_height = int(max_rect_height - completion * (max_rect_height - min_rect_height))
        rect_y = 100  # Fixed top position
    
    # Create frame
    frame = create_press_frame(frame_size, rect_width, rect_height, rect_y)
    
    # Detect press and calculate area
    processed_frame, area = detect_press(frame)
    
    # Determine press state based on area threshold
    if area > area_threshold:
        press_state = "CLOSED"
    else:
        press_state = "OPEN"
    
    # Count presses (when state changes from open to closed)
    if prev_press_state == "OPEN" and press_state == "CLOSED":
        press_count += 1
    
    prev_press_state = press_state
    
    # Add press state and count to the image
    cv2.putText(processed_frame, f"Press: {press_state}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(processed_frame, f"Count: {press_count}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save frame
    all_frames.append(processed_frame)
    areas.append(area)
    press_states.append(press_state)
    
    # Write to video
    out.write(processed_frame)

out.release()

# Display some sample frames and area graph
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Press Machine Simulation Samples", fontsize=16)

# Sample frames
sample_indices = [0, total_frames//8, total_frames//4, 
                  total_frames//2, 3*total_frames//4, 7*total_frames//8, 
                  total_frames-1]

for i, idx in enumerate(sample_indices[:6]):
    row, col = divmod(i, 3)
    axes[row, col].imshow(cv2.cvtColor(all_frames[idx], cv2.COLOR_BGR2RGB))
    axes[row, col].set_title(f"Frame {idx}: {press_states[idx]}")
    axes[row, col].axis('off')

# Area over time plot
axes[2, 0].plot(areas)
axes[2, 0].axhline(y=area_threshold, color='r', linestyle='-')
axes[2, 0].set_title('Contour Area Over Time')
axes[2, 0].set_xlabel('Frame')
axes[2, 0].set_ylabel('Area')
axes[2, 0].grid(True)

# Press state over time
states_numeric = [1 if s == "CLOSED" else 0 for s in press_states]
axes[2, 1].plot(states_numeric, drawstyle='steps-post')
axes[2, 1].set_yticks([0, 1])
axes[2, 1].set_yticklabels(['OPEN', 'CLOSED'])
axes[2, 1].set_title('Press State Over Time')
axes[2, 1].set_xlabel('Frame')
axes[2, 1].grid(True)

# Final count
axes[2, 2].text(0.5, 0.5, f"Total Press Count: {press_count}", 
               horizontalalignment='center', verticalalignment='center',
               fontsize=20)
axes[2, 2].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('press_simulation_results.png', dpi=300)
plt.show()

print(f"Simulation complete. Total press count: {press_count}")
print("Video saved as 'press_simulation.avi'")
print("Analysis saved as 'press_simulation_results.png'")