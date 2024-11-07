import numpy as np
import cv2 
import matplotlib.pyplot as plt
from harris import get_Harris, plot_Harris, save_Harris , tune_blocksize_Harris, tune_thres_Harris
from sift import get_SIFT, plot_SIFT, save_SIFT
from mser import get_MSER, plot_MSER, save_MSER

path = 'my_data/front1.jpg'
name = 'front'

## Harris operator
plot_Harris(path)
save_Harris(path, name)

# different window size (different scaling)
tune_blocksize_Harris(path, name)

# different threshold (lambda_min > threshold)
tune_thres_Harris(path, name)

## SIFT
plot_SIFT(path)
save_SIFT(path, name)

## MSER
plot_MSER(path)
save_MSER(path, name)

