from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import time
import threading
import base64
import re
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Global variables to store the camera feeds and processing results
left_frame = None
right_frame = None
disparity_result = None
depth_result = None
calibration_data = {}
frames_ready = threading.Event()
processing_lock = threading.Lock()
calibration_frames = {'left': [], 'right': []}
is_calibrated = False
processing_active = False

# Camera parameters (will be updated during calibration)
camera_matrices = None
distortion_coeffs = None
rect_maps = None
Q_matrix = None

def estimate_camera_parameters(img_shape, focal_length_mm=4.0, sensor_width_mm=4.8):
    """Estimate camera parameters based on typical phone specs"""
    height, width = img_shape[:2]
    
    # Estimate focal length in pixels
    focal_length_pixels = (width * focal_length_mm) / sensor_width_mm
    
    # Camera matrix (intrinsic parameters)
    camera_matrix = np.array([
        [focal_length_pixels, 0, width/2],
        [0, focal_length_pixels, height/2],
        [0, 0, 1]
    ])
    
    # Assuming minimal distortion for modern phone cameras
    dist_coeffs = np.zeros((5, 1))
    
    return camera_matrix, dist_coeffs

def process_calibration_images():
    """Calibrate cameras from collected frames"""
    global camera_matrices, distortion_coeffs, rect_maps, Q_matrix, is_calibrated, calibration_data
    
    print(f"Starting calibration with {len(calibration_frames['left'])} image pairs")
    
    # Check if we have enough calibration frames
    if len(calibration_frames['left']) < 5 or len(calibration_frames['right']) < 5:
        print("Not enough calibration frames collected")
        return False
    
    # Prepare object points for a 9x6 chessboard
    board_size = (9, 6)
    square_size = 1.0  # arbitrary unit
    
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []
    left_imgpoints = []
    right_imgpoints = []
    
    # Detection criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Process each pair of frames
    for left_img, right_img in zip(calibration_frames['left'], calibration_frames['right']):
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, board_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, board_size, None)
        
        if ret_left and ret_right:
            objpoints.append(objp)
            
            # Refine corner positions
            corners_left_refined = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_refined = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            left_imgpoints.append(corners_left_refined)
            right_imgpoints.append(corners_right_refined)
    
    if not objpoints:
        print("No chessboard detected in any calibration pair")
        return False
    
    img_size = (gray_left.shape[1], gray_left.shape[0])
    
    # Calibrate each camera individually
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, left_imgpoints, img_size, None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, right_imgpoints, img_size, None, None)
    
    # Stereo calibration
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, left_imgpoints, right_imgpoints,
        mtx_left, dist_left, mtx_right, dist_right, img_size,
        criteria=criteria, flags=flags)
    
    # Compute rectification
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, img_size, R, T, alpha=0.0)
    
    # Initialize rectification maps
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        mtx_left, dist_left, R1, P1, img_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        mtx_right, dist_right, R2, P2, img_size, cv2.CV_32FC1)
    
    # Store calibration results
    camera_matrices = (mtx_left, mtx_right)
    distortion_coeffs = (dist_left, dist_right)
    rect_maps = ((map_left_x, map_left_y), (map_right_x, map_right_y))
    Q_matrix = Q
    
    calibration_data = {
        'camera_matrices': camera_matrices,
        'distortion_coeffs': distortion_coeffs,
        'R': R, 'T': T, 'E': E, 'F': F,
        'R1': R1, 'R2': R2,
        'P1': P1, 'P2': P2,
        'Q': Q,
        'img_size': img_size
    }
    
    # Save calibration data to file
    try:
        np.savez('flask_stereo_calibration.npz', **calibration_data)
        print("Calibration data saved to file")
    except Exception as e:
        print(f"Error saving calibration data: {e}")
    
    is_calibrated = True
    return True

def load_calibration_data():
    """Load camera calibration data from file"""
    global camera_matrices, distortion_coeffs, rect_maps, Q_matrix, is_calibrated
    
    try:
        # Load data from file
        data = np.load('flask_stereo_calibration.npz')
        
        # Extract parameters
        mtx_left, mtx_right = data['camera_matrices']
        dist_left, dist_right = data['distortion_coeffs']
        R1, R2 = data['R1'], data['R2']
        P1, P2 = data['P1'], data['P2']
        Q = data['Q']
        img_size = tuple(data['img_size'])
        
        # Initialize rectification maps
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, img_size, cv2.CV_32FC1)
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, img_size, cv2.CV_32FC1)
        
        # Store results
        camera_matrices = (mtx_left, mtx_right)
        distortion_coeffs = (dist_left, dist_right)
        rect_maps = ((map_left_x, map_left_y), (map_right_x, map_right_y))
        Q_matrix = Q
        
        is_calibrated = True
        print("Calibration data loaded from file")
        return True
    
    except (FileNotFoundError, IOError):
        print("No calibration file found")
        return False
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return False

def process_frames():
    """Process frames for depth estimation"""
    global left_frame, right_frame, disparity_result, depth_result, processing_active
    
    print("Frame processing thread started")
    
    while processing_active:
        if frames_ready.wait(timeout=1.0):
            frames_ready.clear()
            
            with processing_lock:
                if left_frame is None or right_frame is None:
                    continue
                
                # Create copies to process
                left_copy = left_frame.copy()
                right_copy = right_frame.copy()
            
            # Ensure frames have the same dimensions
            if left_copy.shape != right_copy.shape:
                right_copy = cv2.resize(right_copy, (left_copy.shape[1], left_copy.shape[0]))
            
            # Apply rectification if calibrated
            if is_calibrated and rect_maps is not None:
                left_rectified = cv2.remap(left_copy, rect_maps[0][0], rect_maps[0][1], cv2.INTER_LINEAR)
                right_rectified = cv2.remap(right_copy, rect_maps[1][0], rect_maps[1][1], cv2.INTER_LINEAR)
            else:
                left_rectified = left_copy
                right_rectified = right_copy
            
            # Draw horizontal lines for rectification check
            for line in range(0, left_rectified.shape[0], 30):
                left_rectified[line, :] = (0, 255, 0)
                right_rectified[line, :] = (0, 255, 0)
            
            # Convert to grayscale
            left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
            
            # Compute disparity map
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16*8,
                blockSize=5,
                P1=8 * 3 * 5**2,
                P2=32 * 3 * 5**2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            
            disparity = stereo.compute(left_gray, right_gray)
            
            # Normalize disparity for visualization
            disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
            
            # Convert disparity to depth if calibrated
            if is_calibrated and Q_matrix is not None:
                # Convert to float32 for reprojection
                disparity_float = disparity.astype(np.float32) / 16.0
                
                # Reproject to 3D
                points_3d = cv2.reprojectImageTo3D(disparity_float, Q_matrix)
                
                # Extract the Z (depth) component
                depth_map = points_3d[:, :, 2]
                
                # Filter out invalid depth values
                mask = disparity_float > disparity_float.min()
                depth_map[~mask] = 0
                
                # Normalize depth for visualization
                valid_depth = depth_map[depth_map > 0]
                if len(valid_depth) > 0:
                    vmin = np.percentile(valid_depth, 5)
                    vmax = np.percentile(valid_depth, 95)
                    depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
                    np.clip((255 * (depth_map - vmin) / (vmax - vmin)), 0, 255, out=depth_normalized, where=depth_map > 0)
                    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
                else:
                    depth_color = np.zeros_like(left_rectified)
            else:
                depth_color = np.zeros_like(left_rectified)
            
            # Store results
            with processing_lock:
                disparity_result = disparity_color
                depth_result = depth_color
    
    print("Frame processing thread stopped")

# Routes
@app.route('/')
def index():
    """Serve the main webpage"""
    return render_template('index.html', is_calibrated=is_calibrated)

@app.route('/camera_feed')
def camera_feed():
    """Serve the camera feed page"""
    return render_template('camera_feed.html')

@app.route('/calibration')
def calibration():
    """Serve the calibration page"""
    return render_template('calibration.html')

@app.route('/results')
def results():
    """Serve the results page"""
    return render_template('results.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    """Receive frames from cameras"""
    global left_frame, right_frame, frames_ready
    
    try:
        data = request.get_json()
        camera_id = data.get('camera')
        image_data = data.get('image')
        
        # Extract base64 data
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        image = np.array(Image.open(BytesIO(image_bytes)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Store frame
        with processing_lock:
            if camera_id == 'left':
                left_frame = image
            elif camera_id == 'right':
                right_frame = image
            
            if left_frame is not None and right_frame is not None:
                frames_ready.set()
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/add_calibration_frame', methods=['POST'])
def add_calibration_frame():
    """Add frame to calibration set"""
    try:
        data = request.get_json()
        camera_id = data.get('camera')
        image_data = data.get('image')
        
        # Extract base64 data
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        image = np.array(Image.open(BytesIO(image_bytes)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Store calibration frame
        if camera_id == 'left':
            calibration_frames['left'].append(image)
        elif camera_id == 'right':
            calibration_frames['right'].append(image)
        
        return jsonify({
            'status': 'success',
            'left_count': len(calibration_frames['left']),
            'right_count': len(calibration_frames['right'])
        })
    
    except Exception as e:
        print(f"Error adding calibration frame: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/start_calibration', methods=['POST'])
def start_calibration():
    """Start the calibration process"""
    success = process_calibration_images()
    
    if success:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Calibration failed'})

@app.route('/reset_calibration', methods=['POST'])
def reset_calibration():
    """Reset the calibration data"""
    global calibration_frames, is_calibrated
    
    calibration_frames = {'left': [], 'right': []}
    is_calibrated = False
    
    return jsonify({'status': 'success'})

@app.route('/get_frame/<frame_type>')
def get_frame(frame_type):
    """Return the requested frame type as an image"""
    global left_frame, right_frame, disparity_result, depth_result
    
    with processing_lock:
        if frame_type == 'left' and left_frame is not None:
            _, buffer = cv2.imencode('.jpg', left_frame)
        elif frame_type == 'right' and right_frame is not None:
            _, buffer = cv2.imencode('.jpg', right_frame)
        elif frame_type == 'disparity' and disparity_result is not None:
            _, buffer = cv2.imencode('.jpg', disparity_result)
        elif frame_type == 'depth' and depth_result is not None:
            _, buffer = cv2.imencode('.jpg', depth_result)
        else:
            # Return a blank image
            blank = np.ones((480, 640, 3), np.uint8) * 128
            _, buffer = cv2.imencode('.jpg', blank)
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/calibration_status')
def calibration_status():
    """Return current calibration status"""
    return jsonify({
        'is_calibrated': is_calibrated,
        'left_count': len(calibration_frames['left']),
        'right_count': len(calibration_frames['right'])
    })

# Start processing thread
# @app.before_first_request
def startup():
    global processing_active
    
    # Try to load existing calibration
    load_calibration_data()
    
    # Start processing thread
    processing_active = True
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()

# Stop processing thread when Flask is shutting down
def shutdown_hook():
    global processing_active
    processing_active = False

# Ensure the templates and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)


if __name__ == '__main__':

    startup()
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        shutdown_hook()