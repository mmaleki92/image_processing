import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
from scipy.ndimage import gaussian_filter

def harris_corner_detector_numpy(image, k=0.04, window_size=3, threshold=0.01):
    """
    Implementation of Harris Corner Detection using NumPy
    
    Parameters:
        image: Input grayscale image
        k: Harris detector free parameter (typically between 0.04-0.06)
        window_size: Size of the window for local maxima detection
        threshold: Threshold for corner detection relative to maximum value
    
    Returns:
        corners: Binary image with detected corners marked
    """
    # Step 1: Compute gradients
    dy, dx = np.gradient(image.astype(np.float32))
    
    # Step 2: Compute products of gradients
    Ixx = dx * dx
    Iyy = dy * dy
    Ixy = dx * dy

    # Step 3: Apply Gaussian smoothing to gradient products
    sigma = 1.0
    Ixx = gaussian_filter(Ixx, sigma)
    Iyy = gaussian_filter(Iyy, sigma)
    Ixy = gaussian_filter(Ixy, sigma)

    # Step 4: Compute Harris response
    det_M = Ixx * Iyy - Ixy * Ixy
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M ** 2)
    
    # Step 5: Threshold and find local maxima
    # Normalize R to [0,1]
    R_normalized = R / R.max() if R.max() > 0 else R
    
    # Apply threshold
    corners = np.zeros_like(R_normalized, dtype=np.uint8)
    corners[R_normalized > threshold] = 1

    # Apply non-maximum suppression
    offset = window_size // 2
    height, width = corners.shape
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if corners[y, x] == 1:
                window = R_normalized[y-offset:y+offset+1, x-offset:x+offset+1]
                if R_normalized[y, x] < np.max(window):
                    corners[y, x] = 0
    
    return corners, R_normalized

def harris_corner_detector_opencv(image, k=0.04, threshold=0.01):
    """
    Implementation of Harris Corner Detection using OpenCV
    
    Parameters:
        image: Input grayscale image
        k: Harris detector free parameter
        threshold: Threshold for corner detection
    
    Returns:
        corners: Binary image with detected corners marked
    """
    # Use OpenCV's cornerHarris
    dst = cv2.cornerHarris(image.astype(np.float32), blockSize=2, ksize=3, k=k)
    
    # Normalize to 0-1 range
    dst_norm = dst / dst.max() if dst.max() > 0 else dst
    
    # Create binary image with thresholded corners
    corners = np.zeros_like(dst_norm, dtype=np.uint8)
    corners[dst_norm > threshold] = 1
    
    return corners, dst_norm

def compare_harris_detectors(image_path):
    """
    Compare the NumPy and OpenCV implementations of Harris Corner Detection
    
    Parameters:
        image_path: Path to the input image
    """
    # Read image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Parameters
    k = 0.04
    threshold = 0.01
    window_size = 3
    
    # Time the NumPy implementation
    start_time = time()
    corners_numpy, response_numpy = harris_corner_detector_numpy(gray, k, window_size, threshold)
    numpy_time = time() - start_time
    print(f"NumPy implementation time: {numpy_time:.4f} seconds")
    
    # Time the OpenCV implementation
    start_time = time()
    corners_opencv, response_opencv = harris_corner_detector_opencv(gray, k, threshold)
    opencv_time = time() - start_time
    print(f"OpenCV implementation time: {opencv_time:.4f} seconds")
    print(f"OpenCV is {numpy_time / opencv_time:.1f}x faster")
    
    # Calculate match percentage
    total_pixels = gray.shape[0] * gray.shape[1]
    matching_pixels = np.sum(corners_numpy == corners_opencv)
    match_percentage = (matching_pixels / total_pixels) * 100
    print(f"Match percentage: {match_percentage:.2f}%")
    
    # Calculate percentage of corner pixels detected by each method
    numpy_corner_pixels = np.sum(corners_numpy == 1)
    opencv_corner_pixels = np.sum(corners_opencv == 1)
    print(f"NumPy detected {numpy_corner_pixels} corner pixels ({numpy_corner_pixels/total_pixels*100:.4f}%)")
    print(f"OpenCV detected {opencv_corner_pixels} corner pixels ({opencv_corner_pixels/total_pixels*100:.4f}%)")
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # NumPy response
    axes[0, 1].imshow(response_numpy, cmap='jet')
    axes[0, 1].set_title('NumPy Harris Response')
    axes[0, 1].axis('off')
    
    # OpenCV response
    axes[0, 2].imshow(response_opencv, cmap='jet')
    axes[0, 2].set_title('OpenCV Harris Response')
    axes[0, 2].axis('off')
    
    # NumPy corners
    img_with_numpy_corners = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    y_coords, x_coords = np.where(corners_numpy == 1)
    for y, x in zip(y_coords, x_coords):
        cv2.circle(img_with_numpy_corners, (x, y), 3, (255, 0, 0), -1)
    axes[1, 0].imshow(img_with_numpy_corners)
    axes[1, 0].set_title('NumPy Detected Corners')
    axes[1, 0].axis('off')
    
    # OpenCV corners
    img_with_opencv_corners = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    y_coords, x_coords = np.where(corners_opencv == 1)
    for y, x in zip(y_coords, x_coords):
        cv2.circle(img_with_opencv_corners, (x, y), 3, (0, 255, 0), -1)
    axes[1, 1].imshow(img_with_opencv_corners)
    axes[1, 1].set_title('OpenCV Detected Corners')
    axes[1, 1].axis('off')
    
    # Comparison (overlay)
    img_comparison = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    # NumPy corners in red
    y_coords, x_coords = np.where(corners_numpy == 1)
    for y, x in zip(y_coords, x_coords):
        cv2.circle(img_comparison, (x, y), 3, (255, 0, 0), -1)
    # OpenCV corners in green
    y_coords, x_coords = np.where(corners_opencv == 1)
    for y, x in zip(y_coords, x_coords):
        cv2.circle(img_comparison, (x, y), 3, (0, 255, 0), -1)
    axes[1, 2].imshow(img_comparison)
    axes[1, 2].set_title('Comparison (Red: NumPy, Green: OpenCV)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('harris_corner_comparison.png')
    plt.show()

if __name__ == "__main__":
	image_path = 'algorithms/SLAM/img_01.png'  
	compare_harris_detectors(image_path)

