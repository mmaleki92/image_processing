import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_hands(image):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Process the image with MediaPipe Hands
    results = hands.process(image_rgb)
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        # Get hand information
        hand_info = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get hand classification (Left or Right)
            hand_type = "Unknown"
            if results.multi_handedness:
                if idx < len(results.multi_handedness):
                    hand_type = results.multi_handedness[idx].classification[0].label
            
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract specific landmark positions (wrist and thumb tip for example)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert normalized coordinates to pixel coordinates
            wrist_x, wrist_y = int(wrist.x * width), int(wrist.y * height)
            thumb_tip_x, thumb_tip_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
            index_tip_x, index_tip_y = int(index_tip.x * width), int(index_tip.y * height)
            
            # Calculate thumb-index distance (for pinch detection example)
            thumb_index_distance = np.sqrt((thumb_tip_x - index_tip_x)**2 + (thumb_tip_y - index_tip_y)**2)
            
            # Store hand info
            hand_info.append({
                "hand_type": hand_type,
                "wrist_position": (wrist_x, wrist_y),
                "thumb_tip_position": (thumb_tip_x, thumb_tip_y),
                "index_tip_position": (index_tip_x, index_tip_y),
                "thumb_index_distance": int(thumb_index_distance)
            })
        
        # Display hand information on the image
        y_position = 30
        for i, info in enumerate(hand_info):
            cv2.putText(image, f"Hand {i+1} ({info['hand_type']})", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_position += 25
            
            cv2.putText(image, f"  Wrist: {info['wrist_position']}", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_position += 25
            
            cv2.putText(image, f"  Thumb tip: {info['thumb_tip_position']}", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_position += 25
            
            cv2.putText(image, f"  Index tip: {info['index_tip_position']}", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_position += 25
            
            cv2.putText(image, f"  Thumb-Index distance: {info['thumb_index_distance']} pixels", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_position += 40
    
    return image

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Process frame for hand detection
        frame = detect_hands(frame)
        
        # Display instructions
        cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('MediaPipe Hands Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()