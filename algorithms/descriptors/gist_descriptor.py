import numpy as np
from scipy import ndimage, signal, fftpack
import matplotlib.pyplot as plt

class GISTDescriptor:
    def __init__(self, num_blocks=4, num_orientations=8, num_scales=4, prefilter=True):
        """
        NumPy/SciPy implementation of GIST descriptor.
        
        Parameters:
        -----------
        num_blocks : int
            Number of blocks to divide the image into (per dimension).
        num_orientations : int
            Number of orientations for the Gabor filters.
        num_scales : int
            Number of scales for the Gabor filters.
        prefilter : bool
            Whether to apply prefiltering to the image.
        """
        self.num_blocks = num_blocks
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.prefilter = prefilter
        
    def _create_gabor_filters(self, img_size):
        """Create Gabor filters in the frequency domain."""
        filters = np.zeros((self.num_orientations, self.num_scales, img_size[0], img_size[1]), dtype=complex)
        
        for i in range(self.num_orientations):
            for j in range(self.num_scales):
                # Calculate parameters
                orientation = np.pi * i / self.num_orientations
                scale = (j + 1) / self.num_scales
                
                # Create the Gabor filter in the frequency domain
                freq_x, freq_y = np.meshgrid(
                    np.fft.fftfreq(img_size[1]),
                    np.fft.fftfreq(img_size[0])
                )
                
                # Rotate frequencies
                rot_freq_x = freq_x * np.cos(orientation) + freq_y * np.sin(orientation)
                rot_freq_y = -freq_x * np.sin(orientation) + freq_y * np.cos(orientation)
                
                # Apply Gabor filter formula
                sigma = 0.35 / scale
                filters[i, j] = np.exp(-(rot_freq_x**2 + rot_freq_y**2) / (2 * sigma**2)) * np.exp(1j * 2 * np.pi * rot_freq_x * scale)
                
                # Remove DC component
                filters[i, j][0, 0] = 0
        
        return filters
        
    def _prefilter_image(self, image):
        """Apply whitening prefilter to image."""
        # Convert to float
        image = image.astype(float)
        
        # Pad image to make it square and power of 2
        max_dim = max(image.shape)
        pad_size = int(2**np.ceil(np.log2(max_dim)))
        
        padded_img = np.zeros((pad_size, pad_size))
        padded_img[:image.shape[0], :image.shape[1]] = image
        
        # Apply FFT
        img_fft = np.fft.fft2(padded_img)
        
        # Create whitening filter
        fx, fy = np.meshgrid(np.fft.fftfreq(padded_img.shape[1]), np.fft.fftfreq(padded_img.shape[0]))
        freq = np.sqrt(fx**2 + fy**2)
        
        # Avoid division by zero
        freq[0, 0] = 1
        
        # Apply filter
        img_fft = img_fft / freq
        img_fft[0, 0] = 0  # Remove DC
        
        # Inverse FFT and crop back to original size
        filtered = np.real(np.fft.ifft2(img_fft))
        filtered = filtered[:image.shape[0], :image.shape[1]]
        
        return filtered
        
    def compute(self, image):
        """Compute GIST descriptor for the given image."""
        # Check if image is grayscale, if not convert it
        if image.ndim == 3:
            image = np.mean(image, axis=2)
        
        # Prefilter if needed
        if self.prefilter:
            image = self._prefilter_image(image)
        
        # Resize image to be square (optional but recommended for GIST)
        target_size = max(image.shape)
        resized_img = np.zeros((target_size, target_size))
        resized_img[:image.shape[0], :image.shape[1]] = image
        
        # Create Gabor filters
        gabor_filters = self._create_gabor_filters((target_size, target_size))
        
        # Process image through Gabor filters
        img_fft = np.fft.fft2(resized_img)
        filtered_responses = np.zeros((self.num_orientations, self.num_scales, target_size, target_size))
        
        for i in range(self.num_orientations):
            for j in range(self.num_scales):
                # Apply filter in frequency domain
                filtered_fft = img_fft * gabor_filters[i, j]
                # Convert back to spatial domain
                response = np.abs(np.fft.ifft2(filtered_fft))
                filtered_responses[i, j] = response
        
        # Divide image into blocks and compute average response in each block
        block_size = target_size // self.num_blocks
        gist_descriptor = np.zeros(self.num_blocks * self.num_blocks * self.num_orientations * self.num_scales)
        
        index = 0
        for i in range(self.num_orientations):
            for j in range(self.num_scales):
                for y in range(self.num_blocks):
                    for x in range(self.num_blocks):
                        # Extract block
                        block = filtered_responses[i, j, y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size]
                        # Compute average response and store in descriptor
                        gist_descriptor[index] = np.mean(block)
                        index += 1
        
        return gist_descriptor
    
    def visualize_responses(self, image):
        """Visualize Gabor filter responses."""
        # Check if image is grayscale
        if image.ndim == 3:
            image = np.mean(image, axis=2)
            
        # Prefilter if needed
        if self.prefilter:
            image = self._prefilter_image(image)
            
        # Create Gabor filters and get responses
        target_size = max(image.shape)
        resized_img = np.zeros((target_size, target_size))
        resized_img[:image.shape[0], :image.shape[1]] = image
        
        gabor_filters = self._create_gabor_filters((target_size, target_size))
        img_fft = np.fft.fft2(resized_img)
        
        # Create visualization grid
        fig, axes = plt.subplots(self.num_scales, self.num_orientations, figsize=(2*self.num_orientations, 2*self.num_scales))
        
        for i in range(self.num_orientations):
            for j in range(self.num_scales):
                filtered_fft = img_fft * gabor_filters[i, j]
                response = np.abs(np.fft.ifft2(filtered_fft))
                
                if self.num_scales == 1:
                    ax = axes[i]
                else:
                    ax = axes[j, i]
                
                ax.imshow(response, cmap='viridis')
                ax.set_title(f'Scale {j+1}, Orient {i+1}')
                ax.axis('off')
                
        plt.tight_layout()
        return fig