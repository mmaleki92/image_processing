import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Configure drawing specifications
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
connection_drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))

# Load sunglasses image
def load_sunglasses(sunglasses_path):
    if not os.path.isfile(sunglasses_path):
        print(f"Error: Sunglasses image not found at {sunglasses_path}")
        return None
        
    # Load image with alpha channel (transparency)
    sunglasses = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)
    
    if sunglasses is None:
        print(f"Error: Could not load sunglasses image from {sunglasses_path}")
        return None
        
    # If image doesn't have alpha channel, add one
    if sunglasses.shape[2] == 3:
        b_channel, g_channel, r_channel = cv2.split(sunglasses)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        sunglasses = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
    print(f"Successfully loaded sunglasses from {sunglasses_path}")
    return sunglasses

# Function to overlay sunglasses on the face
def overlay_sunglasses(image, face_landmarks, sunglasses_img):
    if sunglasses_img is None:
        return image
        
    # Get image dimensions
    image_height, image_width = image.shape[:2]
    
    # Define the key points for sunglasses positioning
    # Left eye outer corner, right eye outer corner, and nose bridge
    # Indices based on MediaPipe Face Mesh
    LEFT_EYE_OUTER = 263  # Left eye outer corner
    RIGHT_EYE_OUTER = 33  # Right eye outer corner
    LEFT_EYE_INNER = 362  # Left eye inner corner
    RIGHT_EYE_INNER = 133  # Right eye inner corner
    NOSE_BRIDGE = 168     # Bridge of nose
    
    # Extract landmark coordinates
    left_eye_outer = (int(face_landmarks.landmark[LEFT_EYE_OUTER].x * image_width),
                     int(face_landmarks.landmark[LEFT_EYE_OUTER].y * image_height))
    
    right_eye_outer = (int(face_landmarks.landmark[RIGHT_EYE_OUTER].x * image_width),
                      int(face_landmarks.landmark[RIGHT_EYE_OUTER].y * image_height))
                      
    left_eye_inner = (int(face_landmarks.landmark[LEFT_EYE_INNER].x * image_width),
                     int(face_landmarks.landmark[LEFT_EYE_INNER].y * image_height))
    
    right_eye_inner = (int(face_landmarks.landmark[RIGHT_EYE_INNER].x * image_width),
                      int(face_landmarks.landmark[RIGHT_EYE_INNER].y * image_height))
                      
    nose_bridge = (int(face_landmarks.landmark[NOSE_BRIDGE].x * image_width),
                  int(face_landmarks.landmark[NOSE_BRIDGE].y * image_height))
    
    # Calculate the width between eyes (add some extra width for style)
    eye_width = int(abs(right_eye_outer[0] - left_eye_outer[0]) * 1.7)
    
    # Calculate the top position for sunglasses (slightly above eyes)
    eye_y = min(left_eye_outer[1], right_eye_outer[1], left_eye_inner[1], right_eye_inner[1])
    top_y = int(eye_y - eye_width * 0.15)  # Move a bit above the eyes
    
    # Calculate sunglasses height proportional to width
    glasses_height = int(eye_width * sunglasses_img.shape[0] / sunglasses_img.shape[1])
    
    # Calculate horizontal center for positioning
    center_x = nose_bridge[0]
    left_x = center_x - eye_width // 2
    
    # Resize sunglasses
    sunglasses_resized = cv2.resize(sunglasses_img, (eye_width, glasses_height))
    
    # Create a region of interest for overlay
    roi = image[top_y:top_y + glasses_height, left_x:left_x + eye_width]
    
    # Check if ROI is within image boundaries
    if roi.shape[0] <= 0 or roi.shape[1] <= 0 or top_y < 0 or left_x < 0:
        return image
        
    # Check if sizes match after clipping
    if roi.shape[0] != sunglasses_resized.shape[0] or roi.shape[1] != sunglasses_resized.shape[1]:
        # Adjust sunglasses size to fit ROI
        sunglasses_resized = cv2.resize(sunglasses_img, (roi.shape[1], roi.shape[0]))
    
    # Extract the alpha channel
    alpha_glasses = sunglasses_resized[:, :, 3] / 255.0
    alpha_glasses = np.expand_dims(alpha_glasses, axis=-1)
    
    # Extract BGR channels
    glasses_bgr = sunglasses_resized[:, :, 0:3]
    
    # Blend sunglasses with the original image based on alpha
    roi_bg = roi * (1 - alpha_glasses)
    roi_fg = glasses_bgr * alpha_glasses
    
    # Combine background and foreground
    try:
        image[top_y:top_y + glasses_height, left_x:left_x + eye_width] = roi_bg + roi_fg
    except ValueError as e:
        print(f"Error overlaying sunglasses: {e}")
    
    return image

# Function to process image and detect face mesh
def process_face_mesh(image, sunglasses_img=None, draw_mesh=False):
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = face_mesh.process(image_rgb)
    
    # Process face mesh if detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh if requested
            if draw_mesh:
                # Draw tesselation (the mesh grid)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw contours (outline of eyes, eyebrows, lips, etc.)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # Draw face oval outline
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            
            # Overlay sunglasses if provided
            if sunglasses_img is not None:
                image = overlay_sunglasses(image, face_landmarks, sunglasses_img)
            
    return image

# Main function for capturing video
def main():
    # Start video capture from webcam
    cap = cv2.VideoCapture(0)
    
    # Ask for sunglasses path
    sunglasses_path = "projects/media_pipe/sunglasses.png" #input("Enter path to sunglasses PNG image (leave empty for no sunglasses): ")
    sunglasses_img = None
    if sunglasses_path:
        sunglasses_img = load_sunglasses(sunglasses_path)
    
    # Option to show mesh
    show_mesh = False #input("Show face mesh? (y/n): ").lower() == 'y'
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame with face mesh
        frame = process_face_mesh(frame, sunglasses_img, show_mesh)
        
        # Display controls on the frame
        cv2.putText(frame, f'Press Q to Quit | M to toggle mesh', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Face Mesh with Sunglasses', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mesh = not show_mesh
            print(f"Face mesh display: {'On' if show_mesh else 'Off'}")
        elif key == ord('s'):
            # Change sunglasses
            sunglasses_path = "projects/media_pipe/sunglasses.png" #input("Enter path to new sunglasses PNG image: ")
            if sunglasses_path:
                new_glasses = load_sunglasses(sunglasses_path)
                if new_glasses is not None:
                    sunglasses_img = new_glasses
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()