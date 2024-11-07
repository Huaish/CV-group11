import numpy as np
import cv2 
import matplotlib.pyplot as plt

def get_SIFT(path):
    img = cv2.imread(path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    # kp will be a list of keypoints 
    # des is a numpy array of shape 
    kp, des = sift.detectAndCompute(gray,None)

    img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

def plot_SIFT(path):
    sift_img = get_SIFT(path)
    plt.imshow(sift_img)
    plt.show()

def save_SIFT(path, name):
    sift_img = get_SIFT(path)
    cv2.imwrite(f'output/{name}_sift_keypoints.jpg', sift_img)

def match_L2_SIFT(path_l, path_r):
    img_l = cv2.imread(path_l)
    gray_l= cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(path_r)
    gray_r= cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()

    # kp will be a list of keypoints 
    # des is a numpy array of shape 
    kp_l, des_l = sift.detectAndCompute(gray_l,None)
    kp_r, des_r = sift.detectAndCompute(gray_r,None)

    # L2-distance
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(des_l,des_r) # Match descriptors
    matches = sorted(matches, key = lambda x:x.distance) # Sort them in the order of their distance
    img = cv2.drawMatches(img_l, kp_l, img_r, kp_r, matches, None, flags=2)

    return img

def match_ratio_SIFT(path_l, path_r, threshold):
    img_l = cv2.imread(path_l)
    gray_l= cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(path_r)
    gray_r= cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()

    # kp will be a list of keypoints 
    # des is a numpy array of shape 
    kp_l, des_l = sift.detectAndCompute(gray_l,None)
    kp_r, des_r = sift.detectAndCompute(gray_r,None)

    # ratio-distance
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_l,des_r, k=2)
    good = [] # Apply ratio test
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])

    img = cv2.drawMatchesKnn(img_l, kp_l, img_r, kp_r, good, None, flags=2) # cv.drawMatchesKnn expects list of lists as matches.

    return img
