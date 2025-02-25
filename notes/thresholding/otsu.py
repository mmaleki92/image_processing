import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image
image_path = "image.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Compute the histogram
hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))


# Normalize histogram (convert counts to probabilities)
hist_norm = hist / hist.sum()

# Compute cumulative sums and cumulative means
cumulative_sum = np.cumsum(hist_norm)
cumulative_mean = np.cumsum(hist_norm * np.arange(256))

# Compute global mean
global_mean = cumulative_mean[-1]

# Compute between-class variance for all possible thresholds
between_class_variance = ((global_mean * cumulative_sum - cumulative_mean) ** 2) / (cumulative_sum * (1 - cumulative_sum))
between_class_variance[np.isnan(between_class_variance)] = 0  # Handle division by zero

# Find the threshold that maximizes between-class variance
otsu_threshold = np.argmax(between_class_variance)

# Apply the threshold to create a binary image
binary_image = (image >= otsu_threshold).astype(np.uint8) * 255

# Plot results
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Grayscale Image")
ax[0].axis("off")

ax[1].plot(hist, color='black')
ax[1].axvline(otsu_threshold, color='red', linestyle="dashed", linewidth=2, label=f'Threshold = {otsu_threshold}')
ax[1].set_title("Histogram with Otsu's Threshold")
ax[1].set_xlabel("Pixel Intensity")
ax[1].set_ylabel("Frequency")
ax[1].legend()

ax[2].imshow(binary_image, cmap='gray')
ax[2].set_title("Binary Image (Otsu's Threshold Applied)")
ax[2].axis("off")

plt.show()

# Return the computed threshold
otsu_threshold
