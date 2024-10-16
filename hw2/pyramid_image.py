import cv2
import numpy as np
from matplotlib import pyplot as plt
from filter import gaussian_filter

# Load the input image
img = cv2.imread('data/task1and2_hybrid_pyramid/1_motorcycle.bmp', cv2.IMREAD_GRAYSCALE)


# List to store pyramid images
pyramid_images = [img]
fft_images = [np.fft.fft2(img)]

# Generate the Gaussian pyramid (e.g., 4 levels)
current_image = img
for _ in range(4):
    smoothed = gaussian_filter(current_image, cutoff_frequency=20)
    downscaled = smoothed[::2, ::2]
    if downscaled.shape[0] < 2 or downscaled.shape[1] < 2:
        break
    pyramid_images.append(downscaled)
    current_image = downscaled
    fft_images.append(np.fft.fft2(downscaled))

# Use matplotlib to plot all the pyramid levels
num_levels = len(pyramid_images)
fig, axes = plt.subplots(2, num_levels, figsize=(15, 5))

for i in range(num_levels):
    axes[0, i].imshow(cv2.cvtColor(pyramid_images[i], cv2.COLOR_BGR2RGB), cmap='gray')
    # axes[0, i].set_title(f'Level {i}')
    axes[0, i].axis('off')

    # Plot the Fourier transform of the image
    fft_image = np.fft.fftshift(fft_images[i])
    axes[1, i].imshow(np.log(1 + np.abs(fft_image)))
    # axes[1, i].set_title(f'FFT Level {i}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()