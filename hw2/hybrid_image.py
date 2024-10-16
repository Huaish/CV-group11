import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
im1 = cv2.imread('data/task1and2_hybrid_pyramid/4_einstein.bmp', cv2.IMREAD_UNCHANGED)
im2 = cv2.imread('data/task1and2_hybrid_pyramid/4_marilyn.bmp', cv2.IMREAD_UNCHANGED)
print(im1.shape)  # im is a numpy array
print(im2.shape)

def check_img(image):
    # Check the number of channels
    if len(image.shape) == 2:  # Grayscale image
        # Convert grayscale to BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # No conversion needed or convert as needed
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Example conversion
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

def gaussian(i, j, cx, cy, hp, sigma):
    coeff = math.exp(-1.0 * ((i - cx)**2 + (j - cy)**2) / (2 * sigma**2))
    if hp:
        return 1 - coeff
    return coeff

def ideal(i, j, cx, cy, hp, D0):
    D = math.sqrt((i - cx)**2 + (j - cy)**2)
    if hp:
        return 0 if D <= D0 else 1
    return 1 if D <= D0 else 0

def GaussianFilter(n_row, n_col, sigma, highPass=True):
    center_x = int(n_row/2) + 1 if n_row % 2 == 1 else int(n_row/2)
    center_y = int(n_col/2) + 1 if n_col % 2 == 1 else int(n_col/2)
    return np.array([[gaussian(i, j, center_x, center_y, highPass, sigma) for j in range(n_col)] for i in range(n_row)])

def IdealFilter(n_row, n_col, cutoff_freq, highPass=True):
    center_x = int(n_row/2) + 1 if n_row % 2 == 1 else int(n_row/2)
    center_y = int(n_col/2) + 1 if n_col % 2 == 1 else int(n_col/2)
    return np.array([[ideal(i, j, center_x, center_y, highPass, cutoff_freq) for j in range(n_col)] for i in range(n_row)])

def filter(image, sigma, isHigh):
    shiftedDFT = fftshift(fft2(image))
    filteredDFT = shiftedDFT * \
        GaussianFilter(
            image.shape[0], image.shape[1], sigma, highPass=isHigh)
    res = ifft2(ifftshift(filteredDFT))
    return np.real(res)

def filter2(image, cuttoff_freq, isHigh):
    shiftedDFT = fftshift(fft2(image))
    filteredDFT = shiftedDFT * \
        IdealFilter(
            image.shape[0], image.shape[1], cuttoff_freq, highPass=isHigh)
    res = ifft2(ifftshift(filteredDFT))
    return np.real(res)

def hybrid_img(high_img, low_img, sigma_h, sigma_l):
    res = filter(high_img, sigma_h, isHigh=True) + \
        filter(low_img, sigma_l, isHigh=False)
    return res

def hybrid_img2(high_img, low_img, cut_h, cut_l):
    res = filter2(high_img, cut_h, isHigh=True) + \
        filter2(low_img, cut_l, isHigh=False)
    return res

def show_img(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()


im1_c = check_img(im1)
im2_c = check_img(im2)
'''
print(im1_c.shape)
print(im2_c.shape)

show_img(im1_c)
show_img(im2_c)
'''
#combine_im = hybrid_img(im1_c , im2_c , 30, 30)
combine_im = hybrid_img2(im1_c , im2_c , 30, 30)

plt.imshow(combine_im, cmap='gray')
plt.show()