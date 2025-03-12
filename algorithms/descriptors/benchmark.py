import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from pathlib import Path
import os

# Import our implementations
from hog_descriptor import HOGDescriptor
from gist_descriptor import GISTDescriptor
from opencv_descriptors import OpenCVHOGDescriptor, OpenCVGISTSimulator

def load_test_images(folder_path=None, sample_size=5):
    """Load test images or create test patterns if no path provided."""
    images = []
    
    if folder_path and os.path.exists(folder_path):
        # Load images from the specified folder
        image_files = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.png"))
        image_files = image_files[:sample_size]  # Take only the specified number of images
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    
    # If no images were loaded, create test patterns
    if not images:
        print("No images found. Creating test patterns...")
        # Create a few test patterns
        # 1. Checkerboard
        checkerboard = np.zeros((256, 256), dtype=np.uint8)
        tile_size = 32
        for i in range(0, 256, tile_size):
            for j in range(0, 256, tile_size):
                if (i//tile_size + j//tile_size) % 2 == 0:
                    checkerboard[i:i+tile_size, j:j+tile_size] = 255
        
        # 2. Gradient
        gradient = np.zeros((256, 256), dtype=np.uint8)
        for i in range(256):
            gradient[:, i] = i
            
        # 3. Circle
        circle = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(circle, (128, 128), 80, 255, -1)
        
        # 4. Lines
        lines = np.zeros((256, 256), dtype=np.uint8)
        for i in range(0, 256, 20):
            cv2.line(lines, (0, i), (255, i), 255, 2)
            cv2.line(lines, (i, 0), (i, 255), 255, 2)
            
        # 5. Random noise
        noise = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        
        images = [checkerboard, gradient, circle, lines, noise]
    
    return images

def benchmark_hog(images):
    """Benchmark HOG implementations."""
    # Initialize descriptors
    custom_hog = HOGDescriptor()
    opencv_hog = OpenCVHOGDescriptor()
    
    results = {
        'custom_time': [],
        'opencv_time': [],
        'descriptor_similarity': [],
        'visualization': []
    }
    
    for img in images:
        # Benchmark custom HOG
        start_time = time.time()
        custom_desc = custom_hog.compute(img)
        custom_time = time.time() - start_time
        results['custom_time'].append(custom_time)
        
        # Benchmark OpenCV HOG
        start_time = time.time()
        opencv_desc = opencv_hog.compute(img)
        opencv_time = time.time() - start_time
        results['opencv_time'].append(opencv_time)
        
        # Compare descriptors (may have different lengths)
        # We'll just compute correlation of the min length
        min_len = min(len(custom_desc), len(opencv_desc.flatten()))
        similarity = np.corrcoef(custom_desc[:min_len], opencv_desc.flatten()[:min_len])[0, 1]
        results['descriptor_similarity'].append(similarity)
        
        # Generate visualizations for comparison
        custom_vis = custom_hog.visualize(img)
        opencv_vis = opencv_hog.visualize(img)
        results['visualization'].append((custom_vis, opencv_vis))
    
    return results

def benchmark_gist(images):
    """Benchmark GIST implementations."""
    # Initialize descriptors
    custom_gist = GISTDescriptor()
    opencv_gist = OpenCVGISTSimulator()
    
    results = {
        'custom_time': [],
        'opencv_time': [],
        'descriptor_similarity': [],
        'visualization': []
    }
    
    for img in images:
        # Benchmark custom GIST
        start_time = time.time()
        custom_desc = custom_gist.compute(img)
        custom_time = time.time() - start_time
        results['custom_time'].append(custom_time)
        
        # Benchmark OpenCV GIST simulation
        start_time = time.time()
        opencv_desc = opencv_gist.compute(img)
        opencv_time = time.time() - start_time
        results['opencv_time'].append(opencv_time)
        
        # Compare descriptors (may have different lengths)
        min_len = min(len(custom_desc), len(opencv_desc))
        similarity = np.corrcoef(custom_desc[:min_len], opencv_desc[:min_len])[0, 1]
        results['descriptor_similarity'].append(similarity)
        
        # For GIST, we'll just visualize the filter responses
        # This would be a figure object for the custom implementation
        # and an image array for the OpenCV implementation
        custom_vis_fig = custom_gist.visualize_responses(img)
        opencv_vis = opencv_gist.visualize_responses(img)
        results['visualization'].append((custom_vis_fig, opencv_vis))
    
    return results

def plot_results(hog_results, gist_results):
    """Plot benchmark results."""
    # Create figure for performance comparison
    plt.figure(figsize=(15, 10))
    
    # Performance plot for HOG
    plt.subplot(2, 2, 1)
    indices = np.arange(len(hog_results['custom_time']))
    width = 0.35
    plt.bar(indices, hog_results['custom_time'], width, label='Custom HOG')
    plt.bar(indices + width, hog_results['opencv_time'], width, label='OpenCV HOG')
    plt.title('HOG Computation Time')
    plt.xlabel('Image Index')
    plt.ylabel('Time (seconds)')
    plt.legend()
    
    # Performance plot for GIST
    plt.subplot(2, 2, 2)
    plt.bar(indices, gist_results['custom_time'], width, label='Custom GIST')
    plt.bar(indices + width, gist_results['opencv_time'], width, label='OpenCV GIST Sim')
    plt.title('GIST Computation Time')
    plt.xlabel('Image Index')
    plt.ylabel('Time (seconds)')
    plt.legend()
    
    # Similarity plot for HOG
    plt.subplot(2, 2, 3)
    plt.plot(hog_results['descriptor_similarity'], 'bo-', label='HOG Descriptor Similarity')
    plt.title('HOG Descriptor Similarity (Correlation)')
    plt.xlabel('Image Index')
    plt.ylabel('Correlation')
    plt.ylim(-1, 1)
    plt.grid(True)
    
    # Similarity plot for GIST
    plt.subplot(2, 2, 4)
    plt.plot(gist_results['descriptor_similarity'], 'ro-', label='GIST Descriptor Similarity')
    plt.title('GIST Descriptor Similarity (Correlation)')
    plt.xlabel('Image Index')
    plt.ylabel('Correlation')
    plt.ylim(-1, 1)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    
    # Create figures for visualization comparison
    # For HOG
    for i, (custom_vis, opencv_vis) in enumerate(hog_results['visualization']):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(custom_vis, cmap='gray')
        plt.title(f'Custom HOG Visualization - Image {i+1}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(opencv_vis, cmap='gray')
        plt.title(f'OpenCV HOG Visualization - Image {i+1}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'hog_visualization_comparison_{i+1}.png')
        plt.close()
    
    # For GIST - this is a bit trickier because custom_vis is a figure object
    for i, (custom_vis_fig, opencv_vis) in enumerate(gist_results['visualization']):
        # Save the custom visualization figure
        custom_vis_fig.savefig(f'gist_custom_visualization_{i+1}.png')
        plt.close(custom_vis_fig)
        
        # Create a new figure for the OpenCV visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(opencv_vis, cmap='gray')
        plt.title(f'OpenCV GIST Visualization - Image {i+1}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'gist_opencv_visualization_{i+1}.png')
        plt.close()

def main(image_folder=None):
    """Main function to run the benchmark."""
    # Load test images
    images = load_test_images(image_folder)
    print(f"Loaded {len(images)} test images.")
    
    # Benchmark HOG
    print("Benchmarking HOG descriptors...")
    hog_results = benchmark_hog(images)
    
    # Benchmark GIST
    print("Benchmarking GIST descriptors...")
    gist_results = benchmark_gist(images)
    
    # Plot results
    print("Plotting results...")
    plot_results(hog_results, gist_results)
    
    # Display average performance metrics
    print("\nHOG Performance Metrics:")
    print(f"Average Custom HOG Time: {np.mean(hog_results['custom_time']):.4f} seconds")
    print(f"Average OpenCV HOG Time: {np.mean(hog_results['opencv_time']):.4f} seconds")
    print(f"Average Descriptor Similarity: {np.mean(hog_results['descriptor_similarity']):.4f}")
    
    print("\nGIST Performance Metrics:")
    print(f"Average Custom GIST Time: {np.mean(gist_results['custom_time']):.4f} seconds")
    print(f"Average OpenCV GIST Time: {np.mean(gist_results['opencv_time']):.4f} seconds")
    print(f"Average Descriptor Similarity: {np.mean(gist_results['descriptor_similarity']):.4f}")
    
    print("\nResults have been saved as PNG files.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark HOG and GIST descriptor implementations")
    parser.add_argument("--folder", type=str, default=None, help="Folder containing test images")
    args = parser.parse_args()

    main("../../images_test")
