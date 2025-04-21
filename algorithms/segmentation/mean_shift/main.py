import sys
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QLabel, QFileDialog, QWidget, QSpinBox, QCheckBox, 
                            QDoubleSpinBox, QGroupBox, QTabWidget, QSlider)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt6.QtCore import Qt, QTimer
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MeanShiftVisualization(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Create the matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Sample points and centers
        self.points = []
        self.centers = []
        self.lines = []
        self.iteration = 0
        self.max_iterations = 0
        
        # Controls for iteration
        control_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_iteration)
        control_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_iteration)
        control_layout.addWidget(self.next_button)
        
        self.iteration_label = QLabel("Iteration: 0")
        control_layout.addWidget(self.iteration_label)
        
        self.layout.addLayout(control_layout)
        
        # Initialize plot
        self.ax = self.figure.add_subplot(111)
    
    def set_data(self, iterations_data):
        """Set the data for visualization"""
        self.iterations_data = iterations_data
        self.iteration = 0
        self.max_iterations = len(iterations_data) - 1
        self.update_plot()
    
    def update_plot(self):
        """Update the plot based on current iteration"""
        if not hasattr(self, 'iterations_data') or len(self.iterations_data) == 0:
            return
            
        self.ax.clear()
        
        # Get current iteration data
        data = self.iterations_data[self.iteration]
        points = data['points']
        shifted_points = data['shifted_points']
        
        # Plot original points
        colors = np.array(['blue'] * len(points))
        
        # Plot vectors showing the shift
        for i in range(len(points)):
            start = points[i]
            end = shifted_points[i]
            self.ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                      head_width=0.3, head_length=0.3, fc='red', ec='red', 
                      length_includes_head=True, alpha=0.3)
        
        # Plot points
        self.ax.scatter(points[:, 0], points[:, 1], c=colors, s=30, alpha=0.6)
        self.ax.scatter(shifted_points[:, 0], shifted_points[:, 1], c='red', s=20, alpha=1.0)
        
        self.ax.set_title(f"Mean Shift Iteration {self.iteration+1}")
        self.ax.set_xlabel("Feature 1")
        self.ax.set_ylabel("Feature 2")
        
        self.iteration_label.setText(f"Iteration: {self.iteration+1}/{self.max_iterations}")
        self.prev_button.setEnabled(self.iteration > 0)
        self.next_button.setEnabled(self.iteration < self.max_iterations-1)
        
        self.canvas.draw()
    
    def next_iteration(self):
        if self.iteration < self.max_iterations - 1:
            self.iteration += 1
            self.update_plot()
    
    def previous_iteration(self):
        if self.iteration > 0:
            self.iteration -= 1
            self.update_plot()

class ImageClusteringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Segmentation with Mean Shift")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.iterations_data = []
        self.sample_points = 500  # Number of sample points for visualization
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main widget and layout
        self.main_widget = QWidget()
        main_layout = QVBoxLayout(self.main_widget)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # First tab - Image Segmentation
        self.tab_segmentation = QWidget()
        segmentation_layout = QVBoxLayout(self.tab_segmentation)
        
        # Control panel
        control_layout = QVBoxLayout()
        
        # First row of controls
        top_control_layout = QHBoxLayout()
        
        # Upload button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        top_control_layout.addWidget(self.upload_button)
        
        # Bandwidth selection
        bandwidth_layout = QHBoxLayout()
        bandwidth_layout.addWidget(QLabel("Bandwidth:"))
        self.bandwidth_spinbox = QDoubleSpinBox()
        self.bandwidth_spinbox.setMinimum(0.1)
        self.bandwidth_spinbox.setMaximum(100.0)
        self.bandwidth_spinbox.setSingleStep(0.5)
        self.bandwidth_spinbox.setValue(10.0)
        bandwidth_layout.addWidget(self.bandwidth_spinbox)
        
        # Auto bandwidth checkbox
        self.auto_bandwidth = QCheckBox("Auto Bandwidth")
        self.auto_bandwidth.setChecked(True)
        self.auto_bandwidth.stateChanged.connect(
            lambda state: self.bandwidth_spinbox.setEnabled(state != Qt.CheckState.Checked.value)
        )
        bandwidth_layout.addWidget(self.auto_bandwidth)
        
        # Update the bandwidth spinbox state
        self.bandwidth_spinbox.setEnabled(not self.auto_bandwidth.isChecked())
        
        top_control_layout.addLayout(bandwidth_layout)
        
        # Process button
        self.process_button = QPushButton("Apply Mean Shift")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        top_control_layout.addWidget(self.process_button)
        
        control_layout.addLayout(top_control_layout)
        
        # Second row of controls - spatial options
        spatial_group = QGroupBox("Spatial Options")
        spatial_layout = QVBoxLayout()
        
        # Include coordinates checkbox
        self.include_coordinates = QCheckBox("Include pixel coordinates (x,y)")
        self.include_coordinates.setChecked(False)
        spatial_layout.addWidget(self.include_coordinates)
        
        # Coordinate weight
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("Spatial weight:"))
        self.spatial_weight = QDoubleSpinBox()
        self.spatial_weight.setMinimum(0.1)
        self.spatial_weight.setMaximum(10.0)
        self.spatial_weight.setSingleStep(0.1)
        self.spatial_weight.setValue(1.0)
        self.spatial_weight.setEnabled(False)
        weight_layout.addWidget(self.spatial_weight)
        spatial_layout.addLayout(weight_layout)
        
        # Connect checkbox to enable/disable weight control
        self.include_coordinates.stateChanged.connect(
            lambda state: self.spatial_weight.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        spatial_group.setLayout(spatial_layout)
        control_layout.addWidget(spatial_group)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        control_layout.addWidget(self.status_label)
        
        segmentation_layout.addLayout(control_layout)
        
        # Image display area
        image_layout = QHBoxLayout()
        
        # Original image
        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel("Original Image:"))
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        original_layout.addWidget(self.original_label)
        image_layout.addLayout(original_layout)
        
        # Processed image
        processed_layout = QVBoxLayout()
        processed_layout.addWidget(QLabel("Segmented Image:"))
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setMinimumSize(400, 400)
        processed_layout.addWidget(self.processed_label)
        image_layout.addLayout(processed_layout)
        
        segmentation_layout.addLayout(image_layout)
        
        # Second tab - Visualization
        self.tab_visualization = QWidget()
        viz_layout = QVBoxLayout(self.tab_visualization)
        
        self.vis_widget = MeanShiftVisualization()
        viz_layout.addWidget(self.vis_widget)
        
        # Add tabs to tab widget
        self.tabs.addTab(self.tab_segmentation, "Image Segmentation")
        self.tabs.addTab(self.tab_visualization, "Iterations Visualization")
        
        main_layout.addWidget(self.tabs)
        
        self.setCentralWidget(self.main_widget)
    
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            # Load the image
            self.original_image = Image.open(file_path)
            
            # Convert the PIL Image to QPixmap for display
            pixmap = self.pil_to_pixmap(self.original_image)
            
            # Display the image
            self.original_label.setPixmap(pixmap.scaled(
                self.original_label.width(), self.original_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio))
            
            self.process_button.setEnabled(True)
            self.status_label.setText("Status: Image loaded")
    
    def process_image(self):
        if self.original_image is None:
            return
        
        self.status_label.setText("Status: Processing...")
        QApplication.processEvents()  # Force UI update
        
        try:
            # Get parameters
            include_coords = self.include_coordinates.isChecked()
            spatial_weight = self.spatial_weight.value()
            
            # Apply Mean Shift clustering
            image_array = np.array(self.original_image)
            h, w, d = image_array.shape
            
            # Reshape the image to be a list of pixels
            image_array_reshaped = image_array.reshape(h * w, d).astype(np.float64)
            
            if include_coords:
                # Create coordinate grid
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                
                # Reshape coordinates to match pixel list
                x_coords = x_coords.reshape(-1, 1)
                y_coords = y_coords.reshape(-1, 1)
                
                # Scale coordinates by specified weight
                # We normalize by image dimensions to make them comparable to color values (0-255)
                x_normalized = x_coords * (255.0 / w) * spatial_weight
                y_normalized = y_coords * (255.0 / h) * spatial_weight
                
                # Combine pixel values with coordinates
                features = np.hstack((image_array_reshaped, x_normalized, y_normalized))
            else:
                features = image_array_reshaped
            
            # Sample some points for visualization
            indices = random.sample(range(len(features)), min(self.sample_points, len(features)))
            sample_features = features[indices]
            
            # Apply Mean Shift
            if self.auto_bandwidth.isChecked():
                bandwidth = estimate_bandwidth(features, quantile=0.2, n_samples=1000)
                if bandwidth <= 0:
                    bandwidth = 10.0  # Fallback if estimation fails
            else:
                bandwidth = self.bandwidth_spinbox.value()
            
            self.status_label.setText(f"Status: Processing with bandwidth={bandwidth:.2f}...")
            QApplication.processEvents()
            
            # Use custom mean shift to get iterations for visualization
            iterations_data = self.custom_mean_shift(sample_features, bandwidth)
            
            # Use sklearn's MeanShift for the actual segmentation for better performance
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(features)
            
            # Get cluster labels and centers
            labels = ms.labels_
            centers = ms.cluster_centers_
            
            # Map each pixel to its cluster center color (the first 3 dimensions are RGB)
            segmented_image = centers[labels][:, :d].reshape(h, w, d)
            
            # Convert to uint8 for image display
            segmented_image = segmented_image.astype(np.uint8)
            
            # Convert segmented image to PIL Image
            self.processed_image = Image.fromarray(segmented_image)
            
            # Convert the PIL Image to QPixmap for display
            pixmap = self.pil_to_pixmap(self.processed_image)
            
            # Display the segmented image
            self.processed_label.setPixmap(pixmap.scaled(
                self.processed_label.width(), self.processed_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio))
            
            # Update the visualization
            self.vis_widget.set_data(iterations_data)
            
            # Switch to the visualization tab
            self.tabs.setCurrentIndex(1)
            
            n_clusters = len(np.unique(labels))
            spatial_mode = "with spatial coordinates" if include_coords else "color only"
            self.status_label.setText(
                f"Status: Image segmented with Mean Shift (bandwidth={bandwidth:.2f}, {n_clusters} clusters, {spatial_mode})"
            )
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def custom_mean_shift(self, X, bandwidth, max_iter=10):
        """
        Custom mean shift implementation that records iterations for visualization.
        Returns a list of dictionaries with points and their shifted positions at each iteration.
        """
        iterations_data = []
        n_samples, n_features = X.shape
        
        # To simplify visualization, limit to 2D (if using color only) or map to 2D using PCA
        if n_features > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X)
        else:
            X_vis = X
            
        # Initialize
        shifted_points = X_vis.copy()
        
        # Main mean shift loop
        for iteration in range(max_iter):
            # Record current state
            iterations_data.append({
                'points': X_vis.copy(),
                'shifted_points': shifted_points.copy()
            })
            
            # Build a nearest neighbors model
            nbrs = NearestNeighbors(radius=bandwidth).fit(X_vis)
            
            # For each point, find neighbors and compute mean shift vector
            for i in range(n_samples):
                # Find neighbors within bandwidth
                indices = nbrs.radius_neighbors([X_vis[i]], bandwidth, return_distance=False)[0]
                
                # Mean shift vector is the mean of the neighbors
                if len(indices) > 0:
                    shifted_points[i] = np.mean(X_vis[indices], axis=0)
            
            # Update X_vis for next iteration
            X_vis = shifted_points.copy()
        
        # Record final state
        iterations_data.append({
            'points': X_vis.copy(),
            'shifted_points': shifted_points.copy()
        })
        
        return iterations_data
    
    def pil_to_pixmap(self, pil_image):
        # Convert PIL Image to QPixmap for display
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        image = QImage(
            pil_image.tobytes(),
            pil_image.width,
            pil_image.height,
            pil_image.width * 3,  # width * 3 channels (RGB)
            QImage.Format.Format_RGB888
        )
        
        return QPixmap.fromImage(image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClusteringApp()
    window.show()
    sys.exit(app.exec())