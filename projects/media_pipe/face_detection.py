import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for short-range detection (2 meters)
    min_detection_confidence=0.5
)

# Function to detect faces in an image
def detect_faces(image):
    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and get face detections
    results = face_detection.process(image_rgb)
    
    # Draw face detections
    if results.detections:
        for detection in results.detections:
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            xmin = int(bboxC.xmin * iw)
            ymin = int(bboxC.ymin * ih)
            width = int(bboxC.width * iw)
            height = int(bboxC.height * ih)
            
            # Draw rectangle with OpenCV
            cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), 
                          (0, 255, 0), 2)  # Green rectangle with 2px thickness
            
            # Add confidence text
            confidence = detection.score[0]
            confidence_text = f"{confidence:.2f}"
            cv2.putText(image, confidence_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            print(f"Detection confidence: {confidence:.2f}")
            
    return image

# Main function
if __name__ == "__main__":
    # Capture video from webcam (0) or video file
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Detect faces
        frame = detect_faces(frame)
        
        # Display the result
        cv2.imshow('Face Detection', frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()