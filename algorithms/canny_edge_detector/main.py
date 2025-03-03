import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge_detector(image_path, low_threshold=50, high_threshold=150):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Unable to load image from path: {image_path}")
        return None
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    
    return edges

def display_image(image, title="Image"):
    # Display the image using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Usage example
image_path = 'images/chicken.png'  # Replace with your image file path
edges = canny_edge_detector(image_path)

if edges is not None:
    display_image(edges, title="Canny Edge Detection")
