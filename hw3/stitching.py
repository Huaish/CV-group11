import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.linalg import svd
import os

def show_keypoints(keypoints1, keypoints2):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Keypoints in Image 1")
    plt.imshow(keypoints1, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Keypoints in Image 2")
    plt.imshow(keypoints2, cmap='gray')
    plt.show()

def show_matches(img1, img2, keypoints1, keypoints2, matches):
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, 
                                  [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _imgIdx=0, _distance=0) for m in matches], 
                                  None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 10))
    plt.title("Feature Matching with Manual Ratio Distance Test")
    plt.imshow(matched_img)
    plt.show()
    
# calculate homography matrix H
def compute_homography(pts1, pts2):
    # Construct matrix A for homography
    A = []
    for i in range(len(pts1)):
        x, y = pts1[i][0], pts1[i][1]
        xp, yp = pts2[i][0], pts2[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)

    # Solve for the homography matrix using SVD
    _, _, V = svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]  # Normalize so that H[2,2] = 1

# RANSAC to find homography matrix H
def ransac_homography(matches, keypoints1, keypoints2, threshold=0.5, max_iterations=2000):
    best_H = None
    max_inliers = 0
    best_inliers = []

    for _ in range(max_iterations):
        # Randomly sample 4 correspondences
        sample_indices = random.sample(range(len(matches)), 4)
        
        pts1 = np.float32([keypoints1[m[0]].pt for m in matches])[sample_indices]
        pts2 = np.float32([keypoints2[m[1]].pt for m in matches])[sample_indices]

        # Compute homography using the sample
        H = compute_homography(pts1, pts2)

        # Count inliers
        inliers = []
        for i, m in enumerate(matches):
            
            p1 = np.array([*keypoints1[m[0]].pt, 1])  # m[0] for the first keypoint index
            p2 = np.array([*keypoints2[m[1]].pt, 1])  # m[1] for the second keypoint index

            # Project point p1 using H
            projected_p2 = np.dot(H, p1)
            projected_p2 /= projected_p2[2]  # Normalize

            # Compute the distance between the projected point and the actual point
            distance = np.linalg.norm(projected_p2[:2] - p2[:2])

            if distance < threshold:
                inliers.append(i)

        # Update best model if the number of inliers is greater than previous
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_H = H
            best_inliers = inliers

    return best_H, best_inliers

def warp_perspective(image, H, size, offset=(0, 0)):
    w, h = size
    offset_x, offset_y = offset
    warped_image = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            # Map (x,y) in the stitched image back to the original image
            src_point = np.linalg.inv(H) @ np.array([x - offset_x, y - offset_y, 1])
            src_point /= src_point[2]

            # Check if the point is within the bounds of the original image, copy the pixel value
            if (
                0 <= src_point[0] < image.shape[1]
                and 0 <= src_point[1] < image.shape[0]
            ):
                warped_image[y, x] = image[int(src_point[1]), int(src_point[0])]
    return warped_image

# Warp image to create panoramic image
def warp(left_img, right_img, H):  
    h_left, w_left = left_img.shape[:2]
    h_right, w_right = right_img.shape[:2]
    
    pts_left = np.float32([[0, 0], [0, h_left], [w_left, h_left], [w_left, 0]])
    pts_right = np.float32([[0, w_right], [0, h_right]])
    
    pts_left_h = np.hstack([pts_left, np.ones((pts_left.shape[0], 1))])
    
    pts_left_transformed = H @ pts_left_h.transpose()
    pts_left_transformed = pts_left_transformed[:2] / pts_left_transformed[2]
    
    pts = np.concatenate((pts_left_transformed, pts_right), axis=1)
    [xmin, ymin] = np.int32(pts.min(axis=1))
    [xmax, ymax] = np.int32(pts.max(axis=1))
   
    translation_dist = [-xmin, -ymin]
    translation_matrix = np.array([
        [1, 0, translation_dist[0]],
        [0, 1, translation_dist[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    H_translation = translation_matrix.dot(H)
    
    size = (xmax - xmin, ymax - ymin)
    
    warped_left = warp_perspective(left_img, H_translation, size)
    warped_right = warp_perspective(right_img, translation_matrix, size)
    
    result = warped_right.copy()
    
    mask_left = warped_left > 0
    result[mask_left] = warped_left[mask_left]
    
    return result

# Interest points detection & feature description by SIFT
def feature_detection(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    mser = cv2.MSER_create()
    regions1, _ = mser.detectRegions(img1)
    regions2, _ = mser.detectRegions(img2)
    
    print("Number of region1 in image 1:", len(regions1))

    keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for p in regions1 for (x, y) in p] 
    keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for p in regions2 for (x, y) in p] 
    
    # keypoints1, descriptors1 = sift.compute(img1, keypoints1)
    # keypoints2, descriptors2 = sift.compute(img2, keypoints2)
    
    
    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    
    print("Number of keypoints in image 1:", len(keypoints1))


    # Draw keypoints on the images
    img1_keypoints = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_keypoints = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    show_keypoints(img1_keypoints, img2_keypoints)
    return keypoints1, descriptors1, keypoints2, descriptors2
    
# Feature matching by SIFT features
def feature_matching(img1, img2, descriptors1, descriptors2, ratio_threshold=0.5):
    # Assuming descriptors1 and descriptors2 are numpy arrays from your SIFT feature detection
    matches = []

    for i, descriptor1 in enumerate(descriptors1):
        distances = []
        for descriptor2 in descriptors2:
            # Compute the Euclidean distance
            distance = np.linalg.norm(descriptor1 - descriptor2)
            distances.append(distance)

        # Sort distances and get the two smallest distances
        distances = np.array(distances)
        sorted_indices = distances.argsort()
        closest_distance = distances[sorted_indices[0]] # ||f1 - f2||
        second_closest_distance = distances[sorted_indices[1]] # ||f1 - f2'||

        # Apply the ratio test
        if closest_distance < ratio_threshold * second_closest_distance:
            # Store the match with the index of descriptor1 and descriptor2
            matches.append((i, sorted_indices[0]))  # (index in descriptors1, index in descriptors2)

    show_matches(img1, img2, keypoints1, keypoints2, matches)
    return matches
    
def read_images(left_image_path, right_image_path, gray=True):
    root_path = "my_data"
    if gray:
        img_l = cv2.imread(os.path.join(root_path, left_image_path), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(os.path.join(root_path, right_image_path), cv2.IMREAD_GRAYSCALE)
    else:
        img_l = cv2.imread(os.path.join(root_path, left_image_path))
        img_r = cv2.imread(os.path.join(root_path, right_image_path))
    
    if img_l is None or img_r is None:
        print("Error loading images")
        exit()
    
    return img_l, img_r


if __name__ == '__main__':
    
    img_l, img_r = read_images("building1.jpg", "building2.jpg")
    
    scale_percent = 10  # 縮小比例，例如50%大小
    width = int(img_l.shape[1] * scale_percent / 100)
    height = int(img_l.shape[0] * scale_percent / 100)
    dim = (width, height)
    print("dimension", dim)
    
    img_l = cv2.resize(img_l, dim, interpolation=cv2.INTER_AREA)
    img_r = cv2.resize(img_r, dim, interpolation=cv2.INTER_AREA)
    
    keypoints1, descriptors1, keypoints2, descriptors2 = feature_detection(img_l, img_r)
    matches = feature_matching(img_l, img_r, descriptors1, descriptors2)
    best_H, inliers = ransac_homography(matches, keypoints1, keypoints2)
    
    # Extract the inlier matches
    inlier_matches = [matches[i] for i in inliers]

    # Convert tuple-based matches to cv2.DMatch objects for visualization
    inlier_matches_for_display = [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _imgIdx=0, _distance=0) for m in inlier_matches]

    # Draw the inlier matches
    img_matches = cv2.drawMatches(img_l, keypoints1, img_r, keypoints2, inlier_matches_for_display, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_l, img_r = read_images("building1.jpg", "building2.jpg", gray=False)
    
    img_l = cv2.resize(img_l, dim, interpolation=cv2.INTER_AREA)
    img_r = cv2.resize(img_r, dim, interpolation=cv2.INTER_AREA)

    # Warp image 2 onto image 1
    panorama = warp(img_l, img_r, best_H)
    
    cv2.imwrite('output/panorama_building.jpg', panorama)
    
