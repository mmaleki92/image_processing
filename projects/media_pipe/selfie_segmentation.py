import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

# Initialize Selfie Segmentation model
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1  # 0 for general model, 1 for landscape model (higher accuracy)
)

def process_segmentation(image, bg_image=None, bg_color=None, threshold=0.1):
    """Process image with MediaPipe Selfie Segmentation and replace background"""
    # Start time for FPS calculation
    start_time = time.time()
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Process the image
    results = selfie_segmentation.process(image_rgb)
    
    # Get segmentation mask
    mask = results.segmentation_mask
    
    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    
    # Create a binary mask based on threshold
    binary_mask = mask > threshold
    
    # Expand mask dimensions to 3 channels
    binary_mask_3d = np.stack((binary_mask,) * 3, axis=-1)
    
    # Create output image based on background options
    output_image = image.copy()
    
    if bg_image is not None:
        # Use a background image
        if bg_image.shape[:2] != image.shape[:2]:
            # Resize background image if needed
            bg_image = cv2.resize(bg_image, (width, height))
        output_image = np.where(binary_mask_3d, image, bg_image)
    
    elif bg_color is not None:
        # Use a solid color background
        bg_image = np.ones(image.shape, dtype=np.uint8)
        bg_image[:] = bg_color  # RGB color as tuple (e.g., (255, 0, 0) for red)
        output_image = np.where(binary_mask_3d, image, bg_image)
    
    else:
        # Default: Set background to grayscale
        bg_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2BGR)
        output_image = np.where(binary_mask_3d, image, bg_image)
    
    # Display mask and FPS
    small_mask = cv2.resize(mask, (width // 4, height // 4))
    small_mask = cv2.normalize(small_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    small_mask = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
    
    # Place mask in top-right corner
    output_image[10:10 + small_mask.shape[0], width - 10 - small_mask.shape[1]:width - 10] = small_mask
    
    # Add FPS and info text
    cv2.putText(output_image, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_image, f"Threshold: {threshold}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return output_image, mask

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting Selfie Segmentation.")
    print("Press 'b' to cycle background options")
    print("Press '+'/'-' to adjust threshold")
    print("Press 'q' to quit")
    
    # Background options
    bg_options = [None, (192, 192, 192), (0, 0, 0), (0, 0, 255)]
    bg_image = None
    bg_option_index = 0
    
    # Background image (if provided)
    try:
        custom_bg = cv2.imread('background.jpg')
        if custom_bg is not None:
            bg_options.append(custom_bg)
            print("Background image loaded successfully")
    except:
        print("No custom background image found")
    
    # Threshold adjustment
    threshold = 0.1
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Get current background option
        bg = bg_options[bg_option_index]
        is_color = isinstance(bg, tuple)
        
        # Process frame with selfie segmentation
        frame, mask = process_segmentation(
            image=frame, 
            bg_image=None if is_color else bg, 
            bg_color=bg if is_color else None, 
            threshold=threshold
        )
        
        # Display background mode text
        if bg is None:
            bg_text = "Mode: Grayscale Background"
        elif isinstance(bg, tuple):
            r, g, b = bg
            bg_text = f"Mode: Solid Color ({r}, {g}, {b})"
        else:
            bg_text = "Mode: Custom Image Background"
        
        cv2.putText(frame, bg_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "B: Change background | +/-: Adjust threshold | Q: Quit", 
                   (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('MediaPipe Selfie Segmentation', frame)
        
        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            # Cycle through background options
            bg_option_index = (bg_option_index + 1) % len(bg_options)
        elif key == ord('+') or key == ord('='):  # '=' is often the unshifted '+' key
            # Increase threshold (make mask more strict)
            threshold = min(0.95, threshold + 0.05)
        elif key == ord('-'):
            # Decrease threshold (make mask less strict)
            threshold = max(0.05, threshold - 0.05)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    selfie_segmentation.close()

if __name__ == "__main__":
    main()