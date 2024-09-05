import cv2
import numpy as np

def is_rectangle(approx, min_area=1000):
    """
    Determine if a contour is a rectangle.
    
    Parameters:
    - approx: The approximated contour
    - min_area: The minimum area to be considered as a valid rectangle
    
    Returns:
    - True if the contour is a rectangle with four corners and an area greater than min_area
    """
    return len(approx) == 4 and cv2.contourArea(approx) > min_area

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the approximated contour is a rectangle
        if is_rectangle(approx):
            # Draw the rectangle on the original frame
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow('Rectangles Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

