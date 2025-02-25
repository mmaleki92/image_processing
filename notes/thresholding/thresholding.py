import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image
image_path = "image.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Compute the histogram
hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))

# Plot the original image and histogram
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Grayscale Image")
ax[0].axis("off")

ax[1].plot(hist, color='black')
ax[1].set_title("Histogram of Pixel Intensities")
ax[1].set_xlabel("Pixel Intensity")
ax[1].set_ylabel("Frequency")

plt.show()
