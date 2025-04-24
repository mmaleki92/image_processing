import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Holistic model
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,  # 0, 1, or 2 (higher is more accurate but slower)
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_holistic(image):
    """Process image with MediaPipe Holistic and draw landmarks"""
    # Start time for FPS calculation
    start_time = time.time()
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Process the image
    results = holistic.process(image_rgb)
    
    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    
    # Draw face landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # Draw left hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
        )
    
    # Draw right hand landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
        )
    
    # Display detection status
    y_pos = 30
    cv2.putText(image, f"FPS: {int(fps)}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    y_pos += 30
    
    cv2.putText(image, f"Face: {'Detected' if results.face_landmarks else 'Not Detected'}", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if results.face_landmarks else (0, 0, 255), 2)
    y_pos += 30
    
    cv2.putText(image, f"Pose: {'Detected' if results.pose_landmarks else 'Not Detected'}", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if results.pose_landmarks else (0, 0, 255), 2)
    y_pos += 30
    
    cv2.putText(image, f"Left Hand: {'Detected' if results.left_hand_landmarks else 'Not Detected'}", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if results.left_hand_landmarks else (0, 0, 255), 2)
    y_pos += 30
    
    cv2.putText(image, f"Right Hand: {'Detected' if results.right_hand_landmarks else 'Not Detected'}", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if results.right_hand_landmarks else (0, 0, 255), 2)
    
    # Extract and display key landmark positions (optional)
    if results.pose_landmarks:
        # Example: Get nose position
        nose_landmark = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
        nose_x = int(nose_landmark.x * width)
        nose_y = int(nose_landmark.y * height)
        cv2.circle(image, (nose_x, nose_y), 5, (255, 0, 0), -1)
    
    return image

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting Holistic Detection. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Process frame with holistic detection
        frame = process_holistic(frame)
        
        # Display instructions
        cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('MediaPipe Holistic Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

if __name__ == "__main__":
    main()