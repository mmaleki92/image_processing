import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def extract_features(image):
    """Extract keypoints and descriptors from image using ORB"""
    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=3000) # 3000 is the number of features
    
    # Find the keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors


def match_features(desc1, desc2):
    """Match features between two images"""
    # Purpose: Creates a Brute-Force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(desc1, desc2)
    
    # Sort them in order of distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches


def estimate_motion(kp1, kp2, matches, K):
    """Estimate camera motion from matched features"""
    # Convert keypoints to numpy arrays
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Find the essential matrix
    # Details: The essential matrix encodes the relative rotation and translation between the two camera positions
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Use only inliers for pose recovery
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in inlier_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in inlier_matches])
    
    # Recover pose (R, t) from essential matrix
    # Details: This gives us the pose of the second camera relative to the first (which is assumed to be at the origin)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, t, pts1, pts2, inlier_matches


def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    """Triangulate 3D points from 2D correspondences"""
    # Create projection matrices
    P1 = np.dot(K, np.hstack((R1, t1)))
    P2 = np.dot(K, np.hstack((R2, t2)))
    
    # Reshape point arrays for triangulation
    pts1 = pts1.T
    pts2 = pts2.T
    
    # Triangulate points
    points_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1, pts2)
    
    # Convert to 3D
    points_3d = points_4d_homogeneous[:3] / points_4d_homogeneous[3]
    
    return points_3d.T


def visualize_results(img1, img2, kp1, kp2, matches, points_3d, R, t):
    """Visualize the results: matched features and 3D reconstruction"""
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot feature matches
    ax1 = fig.add_subplot(131)
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ax1.imshow(match_img)
    ax1.set_title('Feature Matches')
    ax1.axis('off')
    
    # Plot 3D points
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='.', s=1)
    ax2.set_title('3D Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Plot camera poses
    ax3 = fig.add_subplot(133, projection='3d')
    
    # First camera at origin
    camera_center1 = np.array([0, 0, 0])
    
    # Second camera position (transform -R.T @ t)
    camera_center2 = -np.dot(R.T, t).ravel()
    
    # Plot camera positions
    ax3.scatter([camera_center1[0], camera_center2[0]], 
                [camera_center1[1], camera_center2[1]], 
                [camera_center1[2], camera_center2[2]], 
                c='r', marker='o', s=100)
    
    # Visualize camera orientation axes
    axis_length = 0.5
    
    # First camera axes (identity rotation)
    axes1 = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
    
    # Second camera axes
    axes2 = np.dot(R.T, axes1.T).T + camera_center2
    
    # Plot axes for camera 1
    for i, color in enumerate(['r', 'g', 'b']):
        ax3.plot([camera_center1[0], axes1[i][0]], 
                 [camera_center1[1], axes1[i][1]], 
                 [camera_center1[2], axes1[i][2]], color=color, linewidth=2)
    
    # Plot axes for camera 2
    for i, color in enumerate(['r', 'g', 'b']):
        ax3.plot([camera_center2[0], axes2[i][0]], 
                 [camera_center2[1], axes2[i][1]], 
                 [camera_center2[2], axes2[i][2]], color=color, linewidth=2)
    
    ax3.set_title('Camera Poses')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()


def run_slam(image_path1, image_path2):
    """Run SLAM on two images"""
    # Load images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Could not load images from {image_path1} and {image_path2}")
    
    # Extract features
    kp1, desc1 = extract_features(img1)
    kp2, desc2 = extract_features(img2)
    
    print(f"Found {len(kp1)} keypoints in first image")
    print(f"Found {len(kp2)} keypoints in second image")
    
    # Match features
    matches = match_features(desc1, desc2)
    print(f"Found {len(matches)} feature matches")
    
    # Estimate camera matrix (use approximate values if calibration is not available)
    # This assumes a camera with focal length = 1000 and principal point at image center
    height, width = img1.shape[:2]
    focal_length = 1000  # approximate value, should be calibrated for real applications
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])
    
    # First camera pose (set as origin)
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    
    # Estimate second camera pose
    R2, t2, pts1, pts2, good_matches = estimate_motion(kp1, kp2, matches, K)
    print(f"Camera motion computed with {len(good_matches)} inlier matches")
    
    # Triangulate 3D points
    points_3d = triangulate_points(K, R1, t1, R2, t2, pts1, pts2)
    print(f"Triangulated {len(points_3d)} 3D points")
    
    # Filter points that are too far away (outliers)
    distances = np.linalg.norm(points_3d, axis=1)
    median_dist = np.median(distances)
    mask = distances < median_dist * 3  # Filter points that are 3x farther than median
    filtered_points_3d = points_3d[mask]
    print(f"Filtered to {len(filtered_points_3d)} reasonable 3D points")
    
    # Visualize results
    visualize_results(img1, img2, kp1, kp2, good_matches, filtered_points_3d, R2, t2)
    
    return filtered_points_3d, R2, t2


if __name__ == "__main__":
    image1, image2 = "img_01.png", "img_02.png"
    
    run_slam(image1, image2)