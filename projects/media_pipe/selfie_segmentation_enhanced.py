import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Initialize Selfie Segmentation model
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1  # 0 for general model, 1 for landscape model
)

class EffectMode:
    """Enum-like class for different background effects"""
    ORIGINAL = 0
    BLUR = 1
    GRAYSCALE = 2
    SOLID_COLOR = 3
    CUSTOM_IMAGE = 4
    CARTOON = 5
    EDGE_DETECT = 6
    GREEN_SCREEN = 7

def blur_background(image, mask, blur_strength=55):
    """Apply blur effect to background"""
    blurred = cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)
    binary_mask = mask > 0.1
    binary_mask_3d = np.stack((binary_mask,) * 3, axis=-1)
    return np.where(binary_mask_3d, image, blurred)

def cartoon_effect(image):
    """Apply cartoon effect to image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 9, 9
    )
    
    # Convert back to color
    color = cv2.bilateralFilter(image, 9, 300, 300)
    
    # Combine color and edges
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def edge_detection(image):
    """Apply edge detection effect"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_colored

def load_background_images(folder_path="backgrounds"):
    """Load background images from a folder"""
    background_images = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    background_images.append(img)
    
    # Add a default background if none were loaded
    if not background_images:
        # Create a simple gradient background
        bg = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(bg.shape[1]):
            color = int(255 * i / bg.shape[1])
            bg[:, i, :] = (color, 255 - color, 128)
        background_images.append(bg)
    
    return background_images

def process_segmentation(image, effect_mode=EffectMode.ORIGINAL, bg_image=None, bg_color=(0, 128, 0), threshold=0.1):
    """Process image with Selfie Segmentation and apply selected effect"""
    # Start time for FPS calculation
    start_time = time.time()
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Process the image
    results = selfie_segmentation.process(image_rgb)
    
    # Get segmentation mask
    mask = results.segmentation_mask
    
    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    
    # Apply selected effect
    if effect_mode == EffectMode.ORIGINAL:
        # No effect, keep original image
        output_image = image.copy()
    
    elif effect_mode == EffectMode.BLUR:
        # Apply background blur
        output_image = blur_background(image, mask)
    
    elif effect_mode == EffectMode.GRAYSCALE:
        # Convert background to grayscale
        gray_bg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_bg = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)
        binary_mask = mask > threshold
        binary_mask_3d = np.stack((binary_mask,) * 3, axis=-1)
        output_image = np.where(binary_mask_3d, image, gray_bg)
    
    elif effect_mode == EffectMode.SOLID_COLOR:
        # Set background to solid color
        color_bg = np.ones(image.shape, dtype=np.uint8)
        color_bg[:] = bg_color
        binary_mask = mask > threshold
        binary_mask_3d = np.stack((binary_mask,) * 3, axis=-1)
        output_image = np.where(binary_mask_3d, image, color_bg)
    
    elif effect_mode == EffectMode.CUSTOM_IMAGE:
        # Use custom image as background
        if bg_image is not None:
            if bg_image.shape[:2] != image.shape[:2]:
                bg_image = cv2.resize(bg_image, (width, height))
            binary_mask = mask > threshold
            binary_mask_3d = np.stack((binary_mask,) * 3, axis=-1)
            output_image = np.where(binary_mask_3d, image, bg_image)
        else:
            output_image = image.copy()
    
    elif effect_mode == EffectMode.CARTOON:
        # Apply cartoon effect to background only
        cartoon_bg = cartoon_effect(image)
        binary_mask = mask > threshold
        binary_mask_3d = np.stack((binary_mask,) * 3, axis=-1)
        output_image = np.where(binary_mask_3d, image, cartoon_bg)
    
    elif effect_mode == EffectMode.EDGE_DETECT:
        # Apply edge detection to background
        edges_bg = edge_detection(image)
        binary_mask = mask > threshold
        binary_mask_3d = np.stack((binary_mask,) * 3, axis=-1)
        output_image = np.where(binary_mask_3d, image, edges_bg)
    
    elif effect_mode == EffectMode.GREEN_SCREEN:
        # Apply green screen effect (high contrast mask)
        green_screen = np.ones(image.shape, dtype=np.uint8)
        green_screen[:] = (0, 255, 0)  # Green
        binary_mask = mask > threshold
        binary_mask_3d = np.stack((binary_mask,) * 3, axis=-1)
        output_image = np.where(binary_mask_3d, image, green_screen)
    
    else:
        output_image = image.copy()
    
    # Display mask preview in corner
    small_mask = cv2.resize(mask, (width // 6, height // 6))
    small_mask = cv2.normalize(small_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    small_mask = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
    
    # Add white border around mask preview
    border_size = 2
    white_border = np.ones((small_mask.shape[0] + 2*border_size, 
                           small_mask.shape[1] + 2*border_size, 3), 
                          dtype=np.uint8) * 255
    white_border[border_size:-border_size, border_size:-border_size] = small_mask
    small_mask = white_border
    
    # Place mask in corner
    mask_y_offset = 10
    mask_x_offset = width - 10 - small_mask.shape[1]
    output_image[mask_y_offset:mask_y_offset + small_mask.shape[0], 
                mask_x_offset:mask_x_offset + small_mask.shape[1]] = small_mask
    
    # Add FPS and info text
    cv2.putText(output_image, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return output_image, mask

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Load background images
    background_images = load_background_images()
    bg_image_index = 0
    
    # Background solid colors
    bg_colors = [
        (192, 192, 192),  # Light gray
        (0, 0, 0),        # Black
        (0, 0, 255),      # Red (BGR format)
        (0, 255, 0),      # Green
        (255, 0, 0)       # Blue
    ]
    bg_color_index = 0
    
    # Initial settings
    effect_mode = EffectMode.ORIGINAL
    effect_names = [
        "Original", "Blur Background", "Grayscale Background", 
        "Solid Color", "Custom Background", "Cartoon Effect",
        "Edge Detection", "Green Screen"
    ]
    threshold = 0.1
    
    print("Starting Selfie Segmentation.")
    print("Press 'e' to cycle through effects")
    print("Press 'b' to change background image (when in Custom Background mode)")
    print("Press 'c' to change background color (when in Solid Color mode)")
    print("Press '+'/'-' to adjust threshold")
    print("Press 'q' to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Process frame with current effect
        bg_image = background_images[bg_image_index] if effect_mode == EffectMode.CUSTOM_IMAGE else None
        bg_color = bg_colors[bg_color_index] if effect_mode == EffectMode.SOLID_COLOR else (0, 128, 0)
        
        frame, mask = process_segmentation(
            image=frame,
            effect_mode=effect_mode,
            bg_image=bg_image,
            bg_color=bg_color,
            threshold=threshold
        )
        
        # Display effect mode text
        cv2.putText(frame, f"Mode: {effect_names[effect_mode]}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Threshold: {threshold:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "E: Change effect | B/C: Change bg | +/-: Adjust threshold | Q: Quit", 
                   (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('MediaPipe Selfie Segmentation', frame)
        
        # Handle key presses
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            # Cycle through effect modes
            effect_mode = (effect_mode + 1) % len(effect_names)
        elif key == ord('b') and effect_mode == EffectMode.CUSTOM_IMAGE:
            # Cycle through background images
            bg_image_index = (bg_image_index + 1) % len(background_images)
        elif key == ord('c') and effect_mode == EffectMode.SOLID_COLOR:
            # Cycle through background colors
            bg_color_index = (bg_color_index + 1) % len(bg_colors)
        elif key == ord('+') or key == ord('='):
            # Increase threshold
            threshold = min(0.95, threshold + 0.05)
        elif key == ord('-'):
            # Decrease threshold
            threshold = max(0.05, threshold - 0.05)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    selfie_segmentation.close()

if __name__ == "__main__":
    main()