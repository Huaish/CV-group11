import numpy as np
import cv2 
import matplotlib.pyplot as plt

def get_Harris(path):
    img = cv2.imread(path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # cornerHarris(img, blocksize, ksize, k)
    # img - Input image. It should be grayscale and float32 type.
    # blockSize - It is the size of neighbourhood considered for corner detection
    # ksize - Aperture parameter of the Sobel derivative used.
    # k - Harris detector free parameter in the equation.
    dst = cv2.cornerHarris(gray,2,3,0.04)
 
    # result is dilated for marking the corners, not important
    #dst = cv2.dilate(dst,None)
    
    # threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    
    return img

def plot_Harris(path):
    harris_img = get_Harris(path)
    plt.imshow(harris_img)
    plt.show()

def save_Harris(path, name):
    harris_img = get_Harris(path)
    cv2.imwrite(f'output/{name}_harris_keypoints.jpg', harris_img)

def tune_blocksize_Harris(path, name):
    img = cv2.imread(path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # cornerHarris(img, blocksize, ksize, k)
    # img - Input image. It should be grayscale and float32 type.
    # blockSize - It is the size of neighbourhood considered for corner detection
    # ksize - Aperture parameter of the Sobel derivative used.
    # k - Harris detector free parameter in the equation.

    blocksize = (2, 3, 4, 5, 7, 9)
    ksize = (3, 5, 5, 9, 11, 13)
    for i, j in zip(blocksize, ksize):
        dst = cv2.cornerHarris(gray, i, j, 0.04)
        img[dst>0.01*dst.max()]=[0,0,255]
        cv2.imwrite(f'output/{name}_tune_blocksize_{i}_harris_keypoints.jpg', img)

def tune_thres_Harris(path, name):
    img = cv2.imread(path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # cornerHarris(img, blocksize, ksize, k)
    # img - Input image. It should be grayscale and float32 type.
    # blockSize - It is the size of neighbourhood considered for corner detection
    # ksize - Aperture parameter of the Sobel derivative used.
    # k - Harris detector free parameter in the equation.

    for threshold in [0.05, 0.01, 0.005]:
        dst = cv2.cornerHarris(gray,2,3,0.04)
        img[dst>threshold*dst.max()]=[0,0,255]
        cv2.imwrite(f'output/{name}_tune_thres_{threshold}_harris_keypoints.jpg', img)