import sys
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QLabel, QSlider, QFileDialog, QWidget, QSpinBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from sklearn.cluster import KMeans

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
        control_layout = QHBoxLayout()
        
        # Upload button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        control_layout.addWidget(self.upload_button)
        
        # K selection
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("K value:"))
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(2)
        self.k_spinbox.setMaximum(16)
        self.k_spinbox.setValue(5)
        k_layout.addWidget(self.k_spinbox)
        control_layout.addLayout(k_layout)
        
        # Process button
        self.process_button = QPushButton("Apply K-means")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        control_layout.addWidget(self.process_button)
        
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
            
            # Apply K-means clustering
            image_array = np.array(self.original_image)
            h, w, d = image_array.shape
            
            # Reshape the image to be a list of pixels
            image_array_reshaped = image_array.reshape(h * w, d)
            
            # Apply K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(image_array_reshaped)
            
            # Replace each pixel with its centroid value
            clustered_image_array = kmeans.cluster_centers_[labels].reshape(h, w, d)
            
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
            
            self.status_label.setText(f"Status: Image clustered with K={k}")
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
    
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