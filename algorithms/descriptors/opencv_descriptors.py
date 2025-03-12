import cv2
import numpy as np

class OpenCVHOGDescriptor:
    def __init__(self, win_size=(64, 64), block_size=(16, 16), block_stride=(8, 8), 
                 cell_size=(8, 8), nbins=9):
        """
        OpenCV implementation of HOG descriptor.
        
        Parameters:
        -----------
        win_size : tuple
            Detection window size.
        block_size : tuple
            Block size in pixels.
        block_stride : tuple
            Block stride in pixels.
        cell_size : tuple
            Cell size in pixels.
        nbins : int
            Number of bins for the histograms.
        """
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        
    def compute(self, image):
        """Compute HOG descriptor using OpenCV."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Resize if needed
        height, width = gray.shape
        if height != self.hog.winSize[1] or width != self.hog.winSize[0]:
            gray = cv2.resize(gray, self.hog.winSize)
            
        # Compute HOG
        hog_descriptor = self.hog.compute(gray)
        
        return hog_descriptor
        
    def visualize(self, image):
        """
        Generate a visualization of HOG descriptor.
        Note: OpenCV doesn't provide a built-in HOG visualization,
        so we'll use a simple method to draw the gradients.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Resize if needed
        height, width = gray.shape
        if height != self.hog.winSize[1] or width != self.hog.winSize[0]:
            gray = cv2.resize(gray, self.hog.winSize)
            
        # Calculate gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # Create a visualization
        vis_img = np.zeros_like(gray)
        cell_size = self.hog.cellSize[0]
        
        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                # Get magnitudes and angles for current cell
                cell_mag = mag[y:y+cell_size, x:x+cell_size]
                cell_ang = ang[y:y+cell_size, x:x+cell_size]
                
                # Find dominant gradient
                if cell_mag.size > 0:
                    max_idx = np.argmax(cell_mag)
                    max_mag = cell_mag.flat[max_idx]
                    max_ang = cell_ang.flat[max_idx]
                    
                    # Draw line representing dominant gradient
                    center_y = y + cell_size // 2
                    center_x = x + cell_size // 2
                    
                    len_line = min(cell_size, int(max_mag * 0.5))
                    dx = int(np.cos(max_ang) * len_line)
                    dy = int(np.sin(max_ang) * len_line)
                    
                    cv2.line(vis_img, 
                             (center_x - dx, center_y - dy),
                             (center_x + dx, center_y + dy), 
                             255, 1)
        
        return vis_img


class OpenCVGISTSimulator:
    def __init__(self, num_blocks=4, num_orientations=8, num_scales=4):
        """
        OpenCV-based GIST descriptor simulator (OpenCV doesn't have GIST built-in).
        
        Parameters:
        -----------
        num_blocks : int
            Number of blocks to divide the image into (per dimension).
        num_orientations : int
            Number of orientations for the Gabor filters.
        num_scales : int
            Number of scales for the Gabor filters.
        """
        self.num_blocks = num_blocks
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        
        # Pre-compute Gabor kernels
        self.kernels = self._create_gabor_kernels()
        
    def _create_gabor_kernels(self):
        """Create Gabor kernels using OpenCV."""
        kernels = []
        
        for i in range(self.num_scales):
            scale = (i + 1) * 2
            for j in range(self.num_orientations):
                theta = j * np.pi / self.num_orientations
                
                # Create Gabor kernel using OpenCV
                kernel = cv2.getGaborKernel(
                    ksize=(scale*4+1, scale*4+1),
                    sigma=scale,
                    theta=theta,
                    lambd=scale*2,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F
                )
                
                # Normalize the kernel
                kernel /= kernel.sum()
                kernels.append(kernel)
                
        return kernels
        
    def compute(self, image):
        """Compute GIST-like descriptor using OpenCV Gabor filters."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Resize to make the image square
        max_dim = max(gray.shape)
        square_img = np.zeros((max_dim, max_dim), dtype=np.float32)
        square_img[:gray.shape[0], :gray.shape[1]] = gray
        
        # Normalize image
        square_img = square_img.astype(np.float32) / 255.0
        
        # Apply Gabor filters
        responses = []
        for kernel in self.kernels:
            filtered = cv2.filter2D(square_img, cv2.CV_32F, kernel)
            responses.append(filtered)
        
        # Divide into blocks and compute statistics
        block_size = max_dim // self.num_blocks
        descriptor = []
        
        for response in responses:
            for i in range(self.num_blocks):
                for j in range(self.num_blocks):
                    # Extract block
                    block = response[i*block_size:(i+1)*block_size, 
                                     j*block_size:(j+1)*block_size]
                    
                    # Compute mean and append to descriptor
                    descriptor.append(np.mean(block))
        
        return np.array(descriptor)
        
    def visualize_responses(self, image):
        """Visualize Gabor filter responses."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Normalize image
        gray = gray.astype(np.float32) / 255.0
        
        # Create visualization grid
        rows = self.num_scales
        cols = self.num_orientations
        
        # Apply all filters
        vis_responses = []
        for kernel in self.kernels:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            # Normalize for visualization
            filtered = cv2.normalize(filtered, None, 0, 1, cv2.NORM_MINMAX)
            vis_responses.append(filtered)
        
        # Create a grid image for visualization
        cell_size = 200  # Size of each response in the grid
        grid_img = np.zeros((rows * cell_size, cols * cell_size), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(vis_responses):
                    # Resize response to fit the cell
                    response = cv2.resize(vis_responses[idx], (cell_size, cell_size))
                    # Place in grid
                    grid_img[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = response
        
        return grid_img