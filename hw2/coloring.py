import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/task3_colorizing/nativity.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
 
# this is horizontal division
div = h//3
size = img[:div, :]
 
blue = np.zeros((size.shape[0], size.shape[1], 3), dtype=np.uint8)
green = np.zeros((size.shape[0], size.shape[1], 3), dtype=np.uint8)
red = np.zeros((size.shape[0], size.shape[1], 3), dtype=np.uint8)

blue[:, :, 0] = img[:div, :]
green[:, :, 1] = img[div:2*div, :]
red[:, :, 2] = img[2*div:3*div, :]

output = cv2.add(red, green)  # 疊加紅色和綠色
output = cv2.add(output, blue)    # 疊加藍色

plt.imshow(output)
plt.show()