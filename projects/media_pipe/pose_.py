import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 0, 1, or 2 (higher is more accurate but slower)
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_pose(image):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Process the image with MediaPipe Pose
    results = pose.process(image_rgb)
    
    # Check if pose is detected
    if results.pose_landmarks:
        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Extract key landmark positions
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for key points
        # Head landmarks
        nose = (int(landmarks[mp_pose.PoseLandmark.NOSE].x * width), 
                int(landmarks[mp_pose.PoseLandmark.NOSE].y * height))
        
        # Shoulder landmarks
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width), 
                        int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width), 
                         int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height))
        
        # Hip landmarks
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width), 
                   int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height))
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * width), 
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * height))
        
        # Calculate body measurements
        # Shoulder width
        shoulder_width = np.sqrt((right_shoulder[0] - left_shoulder[0])**2 + 
                                (right_shoulder[1] - left_shoulder[1])**2)
        
        # Hip width
        hip_width = np.sqrt((right_hip[0] - left_hip[0])**2 + 
                           (right_hip[1] - left_hip[1])**2)
        
        # Calculate upper body angle (in degrees)
        torso_vector = (right_shoulder[0] - right_hip[0], right_shoulder[1] - right_hip[1])
        angle_rad = np.arctan2(torso_vector[1], torso_vector[0])
        torso_angle = np.degrees(angle_rad)
        
        # Display body measurements on the image
        y_pos = 30
        cv2.putText(image, f"Shoulder Width: {int(shoulder_width)} pixels", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        
        cv2.putText(image, f"Hip Width: {int(hip_width)} pixels", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        
        cv2.putText(image, f"Torso Angle: {int(torso_angle)} degrees", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        
        # Calculate approximate height (in pixels)
        top_head = nose[1] - int(shoulder_width * 0.25)  # Approximate top of head
        height_pixels = int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height) - top_head
        cv2.putText(image, f"Height: {height_pixels} pixels", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw a vertical line to visualize height
        ankle_pos = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * width),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height))
        cv2.line(image, (ankle_pos[0], top_head), ankle_pos, (0, 0, 255), 2)
    
    return image

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Process frame for pose detection
        frame = detect_pose(frame)
        
        # Display instructions
        cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('MediaPipe Pose Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()