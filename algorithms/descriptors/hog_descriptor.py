import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt

class HOGDescriptor:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        NumPy/SciPy implementation of HOG descriptor.
        
        Parameters:
        -----------
        orientations : int
            Number of orientation bins.
        pixels_per_cell : tuple
            Size (in pixels) of a cell.
        cells_per_block : tuple
            Number of cells in each block.
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def _compute_gradients(self, image):
        """Compute x and y gradients of the image using Sobel operator."""
        if image.ndim == 3:
            # Convert to grayscale if it's a color image
            image = np.mean(image, axis=2)
        
        # Compute gradients
        gradient_x = ndimage.sobel(image, axis=1)
        gradient_y = ndimage.sobel(image, axis=0)
        
        # Compute magnitude and orientation
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        orientation = np.arctan2(gradient_y, gradient_x) * (180 / np.pi) % 180
        
        return magnitude, orientation
        
    def _compute_cell_histograms(self, magnitude, orientation):
        """Compute histograms for each cell."""
        sy, sx = magnitude.shape[:2]  # Changed from image.shape to magnitude.shape
        cx, cy = self.pixels_per_cell
        
        n_cellsy = int(sy // cy)
        n_cellsx = int(sx // cx)
        
        # Orientation bin centers
        orientation_bin_centers = np.linspace(0, 180, self.orientations, endpoint=False) + 90/self.orientations
        
        cell_histograms = np.zeros((n_cellsy, n_cellsx, self.orientations))
        
        for y in range(n_cellsy):
            for x in range(n_cellsx):
                # Get magnitudes and orientations for current cell
                cell_mag = magnitude[y*cy:(y+1)*cy, x*cx:(x+1)*cx]
                cell_ori = orientation[y*cy:(y+1)*cy, x*cx:(x+1)*cx]
                
                # For each pixel, find appropriate orientation bin
                for i in range(self.orientations):
                    # Calculate difference of orientation bin
                    orientation_diff = np.minimum(
                        np.abs(cell_ori - orientation_bin_centers[i]),
                        180 - np.abs(cell_ori - orientation_bin_centers[i])
                    )
                    
                    # Weighted vote (weighted by magnitude and inverse distance to bin center)
                    weight = np.exp(-(orientation_diff**2)/(2*((180/self.orientations)**2)))
                    cell_histograms[y, x, i] = np.sum(cell_mag * weight)
        
        return cell_histograms
    
    def _normalize_blocks(self, cell_histograms):
        """Normalize histograms by blocks."""
        n_cellsy, n_cellsx, _ = cell_histograms.shape
        bx, by = self.cells_per_block
        
        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        
        normalized_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, self.orientations))
        
        for y in range(n_blocksy):
            for x in range(n_blocksx):
                block = cell_histograms[y:y+by, x:x+bx, :]
                normalized_blocks[y, x] = block / (np.sqrt(np.sum(block**2) + 1e-6))
        
        return normalized_blocks
    
    def compute(self, image):
        """Compute HOG descriptor for the given image."""
        # Check if image is grayscale, if not convert it
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        
        # Compute gradients
        magnitude, orientation = self._compute_gradients(image)
        
        # Compute cell histograms
        cell_histograms = self._compute_cell_histograms(magnitude, orientation)
        
        # Normalize blocks
        normalized_blocks = self._normalize_blocks(cell_histograms)
        
        # Flatten to get the final descriptor
        hog_descriptor = normalized_blocks.flatten()
        
        return hog_descriptor
    
    def visualize(self, image):
        """Generate a visualization of HOG descriptor."""
        # Compute gradients
        if image.ndim == 3:
            image_gray = np.mean(image, axis=2)
        else:
            image_gray = image
        magnitude, orientation = self._compute_gradients(image_gray)
        
        # Compute cell histograms
        cell_histograms = self._compute_cell_histograms(magnitude, orientation)
        
        # Create a blank image for visualization
        sy, sx = image_gray.shape[:2]
        cx, cy = self.pixels_per_cell
        n_cellsy = int(sy // cy)
        n_cellsx = int(sx // cx)
        
        vis_image = np.zeros((sy, sx))
        
        # Draw orientations
        for y in range(n_cellsy):
            for x in range(n_cellsx):
                for o in range(self.orientations):
                    # Calculate center of the cell
                    center_y = int((y + 0.5) * cy)
                    center_x = int((x + 0.5) * cx)
                    
                    # Calculate orientation angle
                    angle = o * 180 / self.orientations
                    
                    # Calculate line endpoints
                    rad = min(cx, cy) * 0.4
                    line_y = rad * np.sin(np.radians(angle))
                    line_x = rad * np.cos(np.radians(angle))
                    
                    # Draw the line with intensity proportional to histogram value
                    strength = cell_histograms[y, x, o]
                    
                    # Fix for negative dimensions - draw line directly
                    y1 = max(0, min(sy-1, int(center_y - line_y)))
                    y2 = max(0, min(sy-1, int(center_y + line_y)))
                    x1 = max(0, min(sx-1, int(center_x - line_x)))
                    x2 = max(0, min(sx-1, int(center_x + line_x)))
                    
                    # Use Bresenham's line algorithm instead of mgrid
                    points = self._bresenham_line(y1, x1, y2, x2)
                    for py, px in points:
                        if 0 <= py < sy and 0 <= px < sx:
                            vis_image[py, px] += strength
        
        return vis_image
    
    def _bresenham_line(self, y0, x0, y1, x1):
        """Bresenham's line algorithm to generate points in a line."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((y0, x0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
        return points