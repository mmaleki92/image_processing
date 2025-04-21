import sys
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QLabel, QSlider, QFileDialog, QWidget, QSpinBox, QCheckBox, 
                            QDoubleSpinBox, QGroupBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ImageClusteringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Pixel Clustering with K-means")
        self.setGeometry(100, 100, 1000, 600)
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Control panel
        control_layout = QVBoxLayout()
        
        # First row of controls
        top_control_layout = QHBoxLayout()
        
        # Upload button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        top_control_layout.addWidget(self.upload_button)
        
        # K selection
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("K value:"))
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(2)
        self.k_spinbox.setMaximum(16)
        self.k_spinbox.setValue(5)
        k_layout.addWidget(self.k_spinbox)
        top_control_layout.addLayout(k_layout)
        
        # Process button
        self.process_button = QPushButton("Apply K-means")
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
        
        main_layout.addLayout(control_layout)
        
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
        processed_layout.addWidget(QLabel("Clustered Image:"))
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setMinimumSize(400, 400)
        processed_layout.addWidget(self.processed_label)
        image_layout.addLayout(processed_layout)
        
        main_layout.addLayout(image_layout)
        
        self.setCentralWidget(main_widget)
    
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
            # Get the K value
            k = self.k_spinbox.value()
            include_coords = self.include_coordinates.isChecked()
            spatial_weight = self.spatial_weight.value()
            
            # Apply K-means clustering
            image_array = np.array(self.original_image)
            h, w, d = image_array.shape
            
            # Reshape the image to be a list of pixels
            image_array_reshaped = image_array.reshape(h * w, d)
            
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
            
            # Apply K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # The centers will have 3 or 5 dimensions - we only want the color part (first 3)
            centers = kmeans.cluster_centers_[:, :3]
            
            # Replace each pixel with its centroid value
            clustered_image_array = centers[labels].reshape(h, w, d)
            
            # Convert back to uint8
            clustered_image_array = clustered_image_array.astype(np.uint8)
            
            # Convert clustered image to PIL Image
            self.processed_image = Image.fromarray(clustered_image_array)
            
            # Convert the PIL Image to QPixmap for display
            pixmap = self.pil_to_pixmap(self.processed_image)
            
            # Display the processed image
            self.processed_label.setPixmap(pixmap.scaled(
                self.processed_label.width(), self.processed_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio))
            
            spatial_mode = "with spatial coordinates" if include_coords else "color only"
            self.status_label.setText(f"Status: Image clustered with K={k} ({spatial_mode})")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
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