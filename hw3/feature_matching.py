import numpy as np
import cv2 
import matplotlib.pyplot as plt
from sift import get_SIFT, plot_SIFT, match_L2_SIFT, match_ratio_SIFT
from mser import get_MSER, plot_MSER, match_L2_SIFT_MSER, match_ratio_SIFT_MSER, match_ORB_MSER

path_l = 'data/hill1.jpg'
path_r = 'data/hill2.jpg'
name = 'hill'


## SIFT

# L2 distance
match_img = match_L2_SIFT(path_l, path_r)
plt.imshow(match_img)
plt.show()
cv2.imwrite(f'output/{name}_match_l2_sift.jpg', match_img)

# ratio distance (threshold = 0.75)
match_img = match_ratio_SIFT(path_l, path_r, 0.75)
plt.imshow(match_img)
plt.show()
cv2.imwrite(f'output/{name}_match_ratio_sift.jpg', match_img)

# different ratio distance threshold 
for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    match_img = match_ratio_SIFT(path_l, path_r, threshold)
    cv2.imwrite(f'output/{name}_match_ratio_sift_threshold_{threshold}.jpg', match_img)

## SIFT + MSER

# L2 distance
match_img = match_L2_SIFT_MSER(path_l, path_r)
plt.imshow(match_img)
plt.show()
cv2.imwrite(f'output/{name}_match_l2_sift_mser.jpg', match_img)


# ratio distance (threshold = 0.75)
match_img = match_ratio_SIFT_MSER(path_l, path_r, 0.75)
plt.imshow(match_img)
plt.show()
cv2.imwrite(f'output/{name}_match_ratio_sift_mser.jpg', match_img)

# different ratio distance threshold 
for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    match_img = match_ratio_SIFT_MSER(path_l, path_r, threshold)
    cv2.imwrite(f'output/{name}_match_ratio_sift_mser_threshold_{threshold}.jpg', match_img)

## ORB + MSER
match_img = match_ORB_MSER(path_l, path_r)
plt.imshow(match_img)
plt.show()
cv2.imwrite(f'output/{name}_match_orb_mser.jpg', match_img)