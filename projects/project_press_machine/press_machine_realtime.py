import cv2
import numpy as np
import time

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
        
        return frame_with_contour, area, largest_contour
    
    return frame, 0, None

# Parameters
frame_size = (400, 600)  # height, width
rect_width = 200
min_rect_height = 50
max_rect_height = 200
area_threshold = 15000  # Threshold to determine press state

# Initialize variables
press_count = 0
prev_press_state = "OPEN"
cycle = 0

try:
    while True:  # Run indefinitely until Esc key is pressed
        # Calculate the phase of the pressing cycle
        t = time.time()
        phase = (t % 4) / 4  # 4-second cycle
        
        if phase < 0.5:
            # Press going down (rectangle gets bigger)
            completion = phase * 2  # 0 to 1 in first half
            rect_height = int(min_rect_height + completion * (max_rect_height - min_rect_height))
        else:
            # Press going up (rectangle gets smaller)
            completion = (phase - 0.5) * 2  # 0 to 1 in second half
            rect_height = int(max_rect_height - completion * (max_rect_height - min_rect_height))
        
        rect_y = 100  # Fixed top position
        
        # Create frame
        frame = create_press_frame(frame_size, rect_width, rect_height, rect_y)
        
        # Detect press and calculate area
        processed_frame, area, contour = detect_press(frame)
        
        # Determine press state based on area threshold
        if area > area_threshold:
            press_state = "CLOSED"
        else:
            press_state = "OPEN"
        
        # Count presses (when state changes from open to closed)
        if prev_press_state == "OPEN" and press_state == "CLOSED":
            press_count += 1
            cycle = press_count
        
        prev_press_state = press_state
        
        # Add information to the image
        cv2.putText(processed_frame, f"Area: {int(area)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Press: {press_state}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Count: {press_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw threshold line
        if contour is not None:
            # Get bounding box of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw a horizontal line at the threshold
            threshold_line_y = y + h
            if area > area_threshold:
                cv2.line(processed_frame, (0, threshold_line_y), 
                        (frame_size[1], threshold_line_y), (0, 255, 0), 2)
            else:
                cv2.line(processed_frame, (0, y + int(h * area_threshold/area)), 
                        (frame_size[1], y + int(h * area_threshold/area)), (255, 0, 0), 2)
        
        # Show the frame
        cv2.imshow("Press Machine Simulation", processed_frame)
        
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(20)
        if key == 27 or key == ord('q'):  # 27 is the ESC key
            break

finally:
    cv2.destroyAllWindows()
    print(f"Simulation ended. Total press count: {press_count}")