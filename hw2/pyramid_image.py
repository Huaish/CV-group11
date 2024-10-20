import cv2
import numpy as np
from filter import gaussian_filter
from utils import save_image_grid

# Load the input image
img = cv2.imread('data/task1and2_hybrid_pyramid/1_motorcycle.bmp', cv2.IMREAD_GRAYSCALE)

# List to store pyramid images
pyramid_images = [img]
fft_images = [np.fft.fft2(img)]
num_levels = 4
current_image = img

for _ in range(num_levels):
    smoothed = gaussian_filter(current_image, cutoff_frequency=20)
    downscaled = smoothed[::2, ::2]
    if downscaled.shape[0] < 2 or downscaled.shape[1] < 2:
        break
    pyramid_images.append(downscaled)
    current_image = downscaled
    fft_images.append(np.fft.fft2(downscaled))
    
# Save the image grid
num_levels = len(pyramid_images)
images = [cv2.cvtColor(pyramid_image, cv2.COLOR_BGR2RGB) for pyramid_image in pyramid_images]
images += [np.log(1 + np.abs(np.fft.fftshift(fft_image))) for fft_image in fft_images]

save_image_grid(images, [f'Level {i%num_levels+1}' for i in range(num_levels*2)], 2, num_levels, output_path='output/task2_gaussian_pyramid.jpg')