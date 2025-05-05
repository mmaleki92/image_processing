import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Objectron
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Available categories: 'Shoe', 'Chair', 'Cup', 'Camera'
CATEGORY = 'Chair'  # You can change this to any supported category

# Initialize Objectron model
objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.8,
    model_name=CATEGORY#.lower()
)

def process_objectron(image):
    """Process image with MediaPipe Objectron and draw 3D bounding boxes"""
    # Start timer for FPS calculation
    start_time = time.time()
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Process the image
    results = objectron.process(image_rgb)
    
    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    
    # Draw detections
    if results.detected_objects:
        for detected_object in results.detected_objects:
            # Draw landmarks and 3D bounding box
            mp_drawing.draw_landmarks(
                image,
                detected_object.landmarks_2d,
                mp_objectron.BOX_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_box_object_color()
            )
            mp_drawing.draw_axis(
                image, 
                detected_object.rotation,
                detected_object.translation,
                axis_drawing_spec=mp_drawing_styles.get_default_axis_drawing_spec()
            )
            
            # Extract orientation information (rotation matrix)
            rotation = detected_object.rotation
            
            # Extract translation (position)
            translation = detected_object.translation
            
            # Calculate approximate dimensions and position
            landmarks = detected_object.landmarks_2d.landmark
            x_coordinates = [landmark.x for landmark in landmarks]
            y_coordinates = [landmark.y for landmark in landmarks]
            z_coordinates = [landmark.z for landmark in landmarks]
            
            # Get bounding box dimensions
            x_min, x_max = min(x_coordinates), max(x_coordinates)
            y_min, y_max = min(y_coordinates), max(y_coordinates)
            width_ratio = x_max - x_min
            height_ratio = y_max - y_min
            
            # Convert to pixel coordinates
            box_width = int(width_ratio * width)
            box_height = int(height_ratio * height)
            
            # Get center point
            center_x = int((x_min + x_max) * width / 2)
            center_y = int((y_min + y_max) * height / 2)
            
            # Draw additional information near the object
            cv2.putText(image, f"W: {box_width}px, H: {box_height}px", 
                      (center_x - 50, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, (255, 255, 0), 2)
    
    # Display category and FPS info
    cv2.putText(image, f"Category: {CATEGORY}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"FPS: {int(fps)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display detection status
    detection_status = "Detected" if results.detected_objects else "Not Detected"
    cv2.putText(image, f"Status: {detection_status}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               (0, 255, 0) if results.detected_objects else (0, 0, 255), 2)
    
    # Display object count if any detected
    if results.detected_objects:
        cv2.putText(image, f"Count: {len(results.detected_objects)}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"Starting Objectron for {CATEGORY} detection.")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Process frame with 3D object detection
        frame = process_objectron(frame)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('MediaPipe Objectron', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    objectron.close()

if __name__ == "__main__":
    main()