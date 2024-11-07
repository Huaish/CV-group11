import numpy as np
import cv2 
import matplotlib.pyplot as plt

def get_MSER(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mser = cv2.MSER_create(delta=5, min_area=10, max_area=200, max_variation=0.5, min_diversity=0.1)
    regions, _ = mser.detectRegions(gray) # detect regions
    filtered_regions = [p for p in regions if len(p) > 10] # filters

    for region in regions: # draw region
        for point in region:
            cv2.circle(img, (point[0], point[1]), 1, (255, 0, 0), 1)
    
    return img


def plot_MSER(path):
    mser_img = get_MSER(path)
    plt.imshow(mser_img)
    plt.show()

def save_MSER(path, name):
    mser_img = get_MSER(path)
    cv2.imwrite(f'output/{name}_mser_keypoints.jpg', mser_img)

def match_L2_MSER(path_l, path_r):
    img_l = cv2.imread(path_l)
    gray_l= cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(path_r)
    gray_r= cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    
    mser = cv2.MSER_create(delta=5, min_area=10, max_area=200, max_variation=0.5, min_diversity=0.1)

    # kp will be a list of keypoints 
    # des is a numpy array of shape 
    kp_l = mser.detect(gray_l)
    kp_r = mser.detect(gray_r)

    orb = cv2.ORB_create()
    _, des_l = orb.compute(gray_l, kp_l)
    _, des_r = orb.compute(gray_r, kp_r)

    # L2-distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_l,des_r) # Match descriptors
    matches = sorted(matches, key = lambda x:x.distance) # Sort them in the order of their distance
    img = cv2.drawMatches(img_l, kp_l, img_r, kp_r, matches, None, flags=2)

    return img

def match_ratio_MSER(path_l, path_r, threshold):
    img_l = cv2.imread(path_l)
    gray_l= cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(path_r)
    gray_r= cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    
    mser = cv2.MSER_create(delta=5, min_area=10, max_area=200, max_variation=0.5, min_diversity=0.1)

    # kp will be a list of keypoints 
    # des is a numpy array of shape 
    kp_l = mser.detect(gray_l)
    kp_r = mser.detect(gray_r)

    orb = cv2.ORB_create()
    _, des_l = orb.compute(gray_l, kp_l)
    _, des_r = orb.compute(gray_r, kp_r)

    # ratio-distance
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_l,des_r, k=2)
    good = [] # Apply ratio test
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good.append([m])

    img = cv2.drawMatchesKnn(img_l, kp_l, img_r, kp_r, good, None, flags=2) # cv.drawMatchesKnn expects list of lists as matches.

    return img