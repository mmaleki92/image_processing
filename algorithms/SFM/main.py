import cv2
import numpy as np
import open3d as o3d
import os
import time
from scipy.spatial.transform import Rotation

class StructureFromMotion:
    def __init__(self):
        # Parameters for feature detection and matching
        self.min_match_count = 5
        self.feature_extractor = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Camera parameters (you may need to calibrate your camera to get better values)
        # Default values for a typical smartphone camera
        self.K = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ])
        
        # Storage for reconstruction
        self.images = []
        self.image_paths = []
        self.keypoints = []
        self.descriptors = []
        self.camera_poses = []
        self.point_cloud = []
        self.point_cloud_colors = []
        self.camera_indices = []
        self.point_indices = []
        self.points_2d = []
        
    def capture_images_from_ip_camera(self, camera_url, num_images=20, delay=2):
        """Capture images from an IP camera (like a smartphone IP camera app)"""
        print(f"Connecting to camera at {camera_url}...")
        
        # Create output directory
        os.makedirs("captured_images", exist_ok=True)
        
        # Connect to the camera
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            raise Exception("Failed to connect to the IP camera")
        
        print("Successfully connected to camera")
        print(f"Capturing {num_images} images with {delay} seconds interval")
        print("Move around the object to capture different viewpoints")
        
        for i in range(num_images):
            print(f"Capturing image {i+1}/{num_images}...")
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture an image, trying again...")
                time.sleep(1)
                continue
                
            # Save the image
            image_path = f"captured_images/image_{i:03d}.jpg"
            cv2.imwrite(image_path, frame)
            self.image_paths.append(image_path)
            
            # Display the captured frame
            cv2.imshow("Captured Frame", frame)
            cv2.waitKey(100)  # Brief display
            
            # Wait for the specified delay
            time.sleep(delay)
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Successfully captured {len(self.image_paths)} images")
        
    def load_images(self):
        """Load captured images from disk"""
        self.images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read image {path}")
                continue
            self.images.append(img)
        
    def extract_features(self):
        """Extract features from all loaded images"""
        print("Extracting features from images...")
        self.keypoints = []
        self.descriptors = []
        
        for i, img in enumerate(self.images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self.feature_extractor.detectAndCompute(gray, None)
            
            if des is None or len(kp) < 10:
                print(f"Warning: Not enough features detected in image {i}, skipping")
                # Create empty placeholder to maintain indices
                self.keypoints.append([])
                self.descriptors.append(np.array([]))
                continue
                
            self.keypoints.append(kp)
            self.descriptors.append(des)
            
            # Visualization for debugging
            img_with_keypoints = cv2.drawKeypoints(gray, kp, None, 
                                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow(f"Keypoints for image {i}", img_with_keypoints)
            cv2.waitKey(500)
            
        cv2.destroyAllWindows()
        print(f"Extracted features from {len(self.images)} images")
    
    def match_features(self, idx1, idx2):
        """Match features between two images"""
        # Check if descriptors exist
        if len(self.descriptors[idx1]) == 0 or len(self.descriptors[idx2]) == 0:
            return []
            
        matches = self.bf_matcher.knnMatch(self.descriptors[idx1], self.descriptors[idx2], k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        return good_matches
    
    def get_matched_points(self, idx1, idx2, matches):
        """Get matched keypoint coordinates"""
        pts1 = np.float32([self.keypoints[idx1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[idx2][m.trainIdx].pt for m in matches])
        return pts1, pts2
    
    def visualize_matches(self, idx1, idx2, matches):
        """Visualize matches between two images"""
        if not matches:
            print(f"No matches to visualize between images {idx1} and {idx2}")
            return
            
        img1 = self.images[idx1]
        img2 = self.images[idx2]
        
        img_matches = cv2.drawMatches(
            img1, self.keypoints[idx1], 
            img2, self.keypoints[idx2], 
            matches[:min(50, len(matches))], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        cv2.imshow(f"Matches between images {idx1} and {idx2}", img_matches)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    
    def estimate_pose(self, pts1, pts2):
        """Estimate camera pose using essential matrix"""
        # Check if we have enough points
        if pts1.shape[0] < 5 or pts2.shape[0] < 5:
            return None, None, None
            
        # Calculate essential matrix
        E, mask = cv2.findEssentialMat(
            pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        # Check if E is valid
        if E is None or E.shape != (3, 3):
            return None, None, None
        
        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(E, pts2, pts1, self.K, mask=mask)
        
        return R, t, mask

    def triangulate_points(self, idx1, idx2, pts1, pts2, R, t):
        """Triangulate 3D points from matched 2D points and camera poses"""
        # Check if we have valid inputs
        if pts1.size == 0 or pts2.size == 0 or R is None or t is None:
            return np.array([])
        
        # Convert points to proper format for triangulation
        pts1 = np.ascontiguousarray(pts1.T)
        pts2 = np.ascontiguousarray(pts2.T)
        
        # Camera matrices
        P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))  # First camera is at origin
        P2 = np.hstack((R, t))  # Second camera pose
        
        # Apply camera intrinsics
        P1 = np.dot(self.K, P1)
        P2 = np.dot(self.K, P2)
        
        # Ensure correct shapes for triangulation
        if pts1.shape[0] != 2:
            pts1 = pts1[:2, :]
        if pts2.shape[0] != 2:
            pts2 = pts2[:2, :]
            
        # Make sure arrays are in the correct format
        P1 = P1.astype(np.float64)
        P2 = P2.astype(np.float64)
        pts1 = pts1.astype(np.float64)
        pts2 = pts2.astype(np.float64)
        
        try:
            # Triangulate points
            points_4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
            
            # Convert to 3D homogeneous coordinates
            points_3D = points_4D[:3, :] / points_4D[3, :]
            
            return points_3D.T  # Nx3 array of 3D points
            
        except cv2.error as e:
            print(f"Error during triangulation: {e}")
            return np.array([])
    
    def get_point_colors(self, idx1, pts1):
        """Get colors for 3D points from the images"""
        colors = []
        for pt in pts1:
            x, y = int(pt[0]), int(pt[1])
            
            # Ensure points are within image boundaries
            if 0 <= x < self.images[idx1].shape[1] and 0 <= y < self.images[idx1].shape[0]:
                color = self.images[idx1][y, x] / 255.0  # Normalize to 0-1
                colors.append([color[2], color[1], color[0]])  # BGR to RGB
            else:
                colors.append([0.5, 0.5, 0.5])  # Default gray for points outside image
                
        return np.array(colors)
    
    def run_reconstruction(self):
        """Run the structure from motion pipeline"""
        if len(self.images) < 2:
            print("Need at least 2 images for reconstruction")
            return
        
        print(f"Starting reconstruction with {len(self.images)} images")
        
        # Initialize with the first camera at the origin
        self.camera_poses = [np.eye(4)]  # First camera is at origin
        
        # Process image pairs sequentially
        for i in range(len(self.images) - 1):
            print(f"\nProcessing image pair {i}/{i+1}...")
            
            # Match features between consecutive images
            matches = self.match_features(i, i+1)
            
            if len(matches) < self.min_match_count:
                print(f"Not enough matches between images {i} and {i+1}, skipping pair")
                # Add placeholder camera pose to maintain indices
                if i > 0:
                    self.camera_poses.append(self.camera_poses[i])
                else:
                    self.camera_poses.append(np.eye(4))
                continue
            
            print(f"Found {len(matches)} matches")
                
            # Get matched points
            pts1, pts2 = self.get_matched_points(i, i+1, matches)
            
            # Visualize matches
            if len(matches) > 0:
                self.visualize_matches(i, i+1, matches[:min(50, len(matches))])
            
            # Estimate pose
            R, t, mask = self.estimate_pose(pts1, pts2)
            
            if R is None or t is None:
                print(f"Failed to estimate pose between images {i} and {i+1}, skipping pair")
                # Add placeholder camera pose to maintain indices
                if i > 0:
                    self.camera_poses.append(self.camera_poses[i])
                else:
                    self.camera_poses.append(np.eye(4))
                continue
            
            # Apply mask to get inlier points
            if mask is not None:
                mask = mask.ravel() == 1
                pts1_inliers = pts1[mask]
                pts2_inliers = pts2[mask]
                good_matches = [matches[j] for j in range(len(matches)) if j < len(mask) and mask[j]]
            else:
                pts1_inliers = pts1
                pts2_inliers = pts2
                good_matches = matches
            
            print(f"Using {len(pts1_inliers)} point pairs after RANSAC filtering")
                
            # Calculate relative pose
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = t.ravel()
            
            # Calculate absolute pose
            if i > 0:
                T_abs = self.camera_poses[i] @ T_rel
            else:
                T_abs = T_rel
                
            self.camera_poses.append(T_abs)
            
            # Triangulate points
            points_3D = self.triangulate_points(i, i+1, pts1_inliers, pts2_inliers, R, t)
            
            if len(points_3D) == 0:
                print(f"Failed to triangulate points between images {i} and {i+1}")
                continue
                
            print(f"Triangulated {len(points_3D)} 3D points")
                
            # Get point colors
            colors = self.get_point_colors(i, pts1_inliers)
            
            # Store the 3D points and their colors
            self.point_cloud.extend(points_3D)
            self.point_cloud_colors.extend(colors)
            
            # Store correspondences for bundle adjustment
            for j in range(len(pts1_inliers)):
                self.camera_indices.append(i)
                self.point_indices.append(len(self.point_cloud) - len(pts1_inliers) + j)
                self.points_2d.append(pts1_inliers[j])
                
                self.camera_indices.append(i+1)
                self.point_indices.append(len(self.point_cloud) - len(pts1_inliers) + j)
                self.points_2d.append(pts2_inliers[j])
            
            print(f"Processed image pair {i}/{i+1}, added {len(points_3D)} points")
        
        print("Structure from Motion reconstruction completed!")
    
    def filter_point_cloud(self):
        """Filter the point cloud to remove outliers"""
        if not self.point_cloud:
            print("No points to filter")
            return
            
        print("Filtering point cloud...")
            
        # Convert to numpy arrays
        points = np.array(self.point_cloud)
        colors = np.array(self.point_cloud_colors)
        
        if len(points) == 0:
            print("No points to filter after conversion")
            return
            
        # Remove points with NaN or Inf values
        valid_indices = np.all(np.isfinite(points), axis=1)
        points = points[valid_indices]
        colors = colors[valid_indices] if len(colors) > 0 else []
        
        if len(points) == 0:
            print("No points left after removing NaN/Inf values")
            return
        
        # Remove points that are too far from the origin (outliers)
        distances = np.linalg.norm(points, axis=1)
        median_distance = np.median(distances)
        mask = distances < median_distance * 5  # Keep points within 5x median distance
        
        filtered_points = points[mask]
        filtered_colors = colors[mask] if len(colors) > 0 else []
        
        print(f"Filtered point cloud: {len(filtered_points)} points remaining out of {len(points)}")
        
        self.point_cloud = filtered_points.tolist()
        self.point_cloud_colors = filtered_colors.tolist() if len(filtered_colors) > 0 else []
    
    def visualize_reconstruction(self):
        """Visualize the 3D reconstruction"""
        if not self.point_cloud or len(self.point_cloud) == 0:
            print("No points to visualize")
            return
            
        print("Creating visualization...")
        
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        
        if len(self.point_cloud_colors) > 0 and len(self.point_cloud_colors) == len(self.point_cloud):
            pcd.colors = o3d.utility.Vector3dVector(self.point_cloud_colors)
        else:
            # Use default colors if point colors are not available
            default_color = np.array([[1.0, 0.7, 0.0] for _ in range(len(self.point_cloud))])
            pcd.colors = o3d.utility.Vector3dVector(default_color)
        
        # Create camera frustum geometries
        camera_frustums = []
        for pose in self.camera_poses:
            frustum = self.create_camera_frustum(pose)
            camera_frustums.append(frustum)
        
        # Create coordinate system
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        
        # Statistical outlier removal for better visualization
        try:
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = cl
            print(f"Removed additional outliers for visualization, {len(ind)} points remaining")
        except Exception as e:
            print(f"Could not perform statistical outlier removal: {e}")
        
        # Visualize
        print("Displaying 3D visualization. Use mouse to rotate, shift+click to pan, scroll to zoom.")
        print("Close the visualization window to exit.")
        o3d.visualization.draw_geometries([pcd, coordinate_frame] + camera_frustums, 
                                          window_name="Structure from Motion Reconstruction",
                                          width=1200, height=800,
                                          point_show_normal=False)
    
    def create_camera_frustum(self, pose, scale=0.1):
        """Create a camera frustum visualization for a given pose"""
        # Check if pose is valid
        if pose is None or pose.shape != (4, 4):
            # Return empty line set
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
            line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 0]]))
            return line_set
            
        # Camera frustum points in camera coordinates
        frustum_points = np.array([
            [0, 0, 0],  # Camera center
            [1, 1, 1],  # Top-right
            [1, -1, 1],  # Bottom-right
            [-1, -1, 1],  # Bottom-left
            [-1, 1, 1]   # Top-left
        ]) * scale
        
        # Transform points to world coordinates
        frustum_points_world = []
        for pt in frustum_points:
            # Convert to homogeneous coordinates
            pt_h = np.append(pt, 1)
            # Apply transformation
            pt_world = pose @ pt_h
            frustum_points_world.append(pt_world[:3])
        
        frustum_points_world = np.array(frustum_points_world)
        
        # Create lines connecting points to represent frustum
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]    # Lines connecting corners
        ]
        
        # Create line set geometry
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(frustum_points_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set colors (blue for camera frustums)
        colors = [[0, 0, 1] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
        
    def save_point_cloud(self, filename="reconstruction.ply"):
        """Save the reconstructed point cloud to a PLY file"""
        if not self.point_cloud or len(self.point_cloud) == 0:
            print("No points to save")
            return
            
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
        
        if len(self.point_cloud_colors) == len(self.point_cloud):
            pcd.colors = o3d.utility.Vector3dVector(self.point_cloud_colors)
        
        # Save to file
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Point cloud saved to {filename}")




def main():
    """Main function to run Structure from Motion"""
    # Initialize SfM
    sfm = StructureFromMotion()
    
    # Get camera URL from user
    camera_url = "http://172.25.194.78:8080/video"#input("Enter your IP camera URL (e.g., http://172.25.194:8080/video): ")
    
    try:
        # Capture images from IP camera
        num_images = 100#int(input("How many images to capture (default: 20): ") or "20")
        delay = 0.01#int(input("Delay between captures in seconds (default: 2): ") or "2")
        sfm.capture_images_from_ip_camera(camera_url, num_images, delay)
        
        # Load the captured images
        sfm.load_images()
        
        # Process the images
        sfm.extract_features()
        sfm.run_reconstruction()
        sfm.filter_point_cloud()
        
        # Visualize the results
        sfm.visualize_reconstruction()
        
        # Save the point cloud
        save_option = input("Do you want to save the point cloud? (y/n): ")
        if save_option.lower() == 'y':
            filename = input("Enter filename (default: reconstruction.ply): ") or "reconstruction.ply"
            sfm.save_point_cloud(filename)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nAlternatively, you can use pre-captured images by placing them in a 'captured_images' folder")
        
        # Ask if user wants to continue with existing images
        choice = input("Do you have existing images to use? (y/n): ")
        if choice.lower() == 'y':
            image_dir = "captured_images"
            if not os.path.exists(image_dir):
                print(f"Error: Directory '{image_dir}' does not exist")
                return
                
            sfm.image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(sfm.image_paths) < 2:
                print(f"Error: Not enough images found in '{image_dir}' directory")
                return
                
            print(f"Found {len(sfm.image_paths)} images")
            
            # Process the images
            sfm.load_images()
            sfm.extract_features()
            sfm.run_reconstruction()
            sfm.filter_point_cloud()
            sfm.visualize_reconstruction()
            
            # Save the point cloud
            save_option = input("Do you want to save the point cloud? (y/n): ")
            if save_option.lower() == 'y':
                filename = input("Enter filename (default: reconstruction.ply): ") or "reconstruction.ply"
                sfm.save_point_cloud(filename)


if __name__ == "__main__":
    main()