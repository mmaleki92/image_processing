import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh with iris
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Create face mesh with iris refinement enabled
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # This enables iris detection
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Iris landmark indices
# Mediapipe provides landmarks 468-477 for the irises
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Function to detect and visualize irises only (no face mesh)
def detect_irises_only(image):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Process the image
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract all mesh points
            mesh_points = np.array([
                [int(landmark.x * width), int(landmark.y * height)]
                for landmark in face_landmarks.landmark
            ])
            
            # Draw left iris
            left_iris_points = [mesh_points[idx] for idx in LEFT_IRIS]
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(np.array(left_iris_points))
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            cv2.circle(image, center_left, int(l_radius), (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw right iris
            right_iris_points = [mesh_points[idx] for idx in RIGHT_IRIS]
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(np.array(right_iris_points))
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(image, center_right, int(r_radius), (0, 255, 0), 2, cv2.LINE_AA)
            
            # Optional: Draw iris landmarks as points
            for point in left_iris_points:
                cv2.circle(image, point, 1, (0, 0, 255), -1, cv2.LINE_AA)
            for point in right_iris_points:
                cv2.circle(image, point, 1, (0, 0, 255), -1, cv2.LINE_AA)
                
            # Add text labels
            cv2.putText(image, f'Left iris center: {center_left}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, f'Right iris center: {center_right}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    return image

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Process frame for iris detection only
        frame = detect_irises_only(frame)
        
        # Display instructions
        cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('MediaPipe Iris Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()