import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-img", type=str, default="data/hill1.JPG")
    parser.add_argument("--right-img", type=str, default="data/hill2.JPG")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


def read_image(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb, img_gray


#  ================== 1. Interest points detection & feature description by SIFT  ==================
def sift_point(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def show_keypoints(img_gray, img_rgb, kp, output_path=None):
    img_kp = cv2.drawKeypoints(
        img_gray, kp, img_rgb.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img_kp, cv2.COLOR_RGB2BGR))
        print(f"Keypoints image saved to {output_path}")

    return img_kp


#  ================== 2. Feature matching by SIFT features  ==================
def matcher_with_cv(kp1, kp2, des1, des2, threshold=0.5):

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches


def matcher(kp1, kp2, des1, des2, threshold=0.5):
    good_matches = []

    # Loop through each descriptor in des1
    for i, d1 in enumerate(des1):
        # Compute the Euclidean distance to all descriptors in des2
        distances = np.linalg.norm(des2 - d1, axis=1)

        # Find the two closest matches
        sorted_indices = np.argsort(distances)
        best_match_idx = sorted_indices[0]
        second_best_match_idx = sorted_indices[1]

        # Apply the ratio test
        if distances[best_match_idx] < threshold * distances[second_best_match_idx]:
            # Save the keypoints that passed the ratio test
            good_matches.append(list(kp1[i].pt + kp2[best_match_idx].pt))

    # Convert matches to a numpy array for consistency
    matches = np.array(good_matches)
    return matches


# TODO: use different matcher and feature: ratio test, cross-checking, RANSAC, MSER, etc.


def show_matches(img_combined, matches, output_path=None):
    import matplotlib.cm as cm

    offset = img_combined.shape[1] // 2
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    plt.imshow(img_combined, cmap="gray")

    # Get a colormap (e.g., 'viridis') and generate colors based on the number of matches
    colormap = plt.get_cmap("viridis")

    for i in range(matches.shape[0]):
        color = colormap(i)  # Color from the colormap based on index

        # Plot keypoints on both images with the gradient color
        ax.plot(matches[i, 0], matches[i, 1], "o", color=color, markersize=1)
        ax.plot(matches[i, 2] + offset, matches[i, 3], "o", color=color, markersize=1)

        # Draw a line connecting the matched keypoints
        ax.plot(
            [matches[i, 0], matches[i, 2] + offset],
            [matches[i, 1], matches[i, 3]],
            color=color,
            linewidth=0.8,
        )

    plt.axis("off")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Matches image saved to {output_path}")


#  ================== 3. RANSAC to find homography matrix H  ==================


def homography(points1, points2):
    A = []
    for i in range(4):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]


def ransac_homography(matches, iterations=1000, threshold=0.5):
    points_in_img1 = matches[:, :2]
    points_in_img2 = matches[:, 2:]

    max_inliers = 0
    best_H = None

    for _ in range(iterations):
        # random sample 4 pairs of points
        idx = np.random.choice(len(points_in_img1), 4, replace=False)
        sample_points1 = points_in_img1[idx]
        sample_points2 = points_in_img2[idx]

        # Compute homography matrix H
        H = homography(sample_points1, sample_points2)

        # Avoid degenerate solutions
        if np.linalg.matrix_rank(H) < 3:
            continue

        # Calculate inliers
        inliers = []
        for i in range(len(points_in_img1)):
            pt1 = np.append(points_in_img1[i], 1)
            projected_pt2 = H @ pt1
            projected_pt2 /= projected_pt2[2]

            # Calculate the error
            error = np.linalg.norm(projected_pt2[:2] - points_in_img2[i])
            if error < threshold:
                inliers.append(i)

        # Update the best H if we found more inliers
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_H = H.copy()

    print(f"Found {max_inliers} inliers out of {len(matches)} matches")
    return inliers, best_H


# ================== 4. Warp image to create panoramic image (stitch img) ==================


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


def warp(left, right, H):
    hl, wl = left.shape[:2]
    hr, wr = right.shape[:2]

    # Get the corners of the right image and transform them using H
    corners = np.array([[0, 0], [0, hr - 1], [wr - 1, hr - 1], [wr - 1, 0]])
    homo_corners = np.hstack([corners, np.ones((corners.shape[0], 1))])

    # perspective transform
    transformed_points = H @ homo_corners.T
    transformed_corners = transformed_points[:2] / transformed_points[2]

    # Calculate the size of the stitched image
    min_x = min(np.min(transformed_corners[0]), 0)
    min_y = min(np.min(transformed_corners[1]), 0)
    max_x = max(np.max(transformed_corners[0]), wl)
    max_y = max(np.max(transformed_corners[1]), hl)

    # Calculate the translation matrix
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    # Size of the output image
    size = (int(max_x - min_x), int(max_y - min_y))

    # Warp the right image to align with the left image's coordinate system
    warped_left = warp_perspective(left, T @ H, size)
    warped_right = warp_perspective(right, T, size)

    # === Compare with cv2.warpPerspective ===
    # warped_left = cv2.warpPerspective(left, T @ H, size)
    # warped_right = cv2.warpPerspective(right, T, size)

    return warped_left, warped_right


def stitch_image(left, right, H, blend=False):
    warped_left, warped_right = warp(left, right, H)

    # Initialize the stitched image with the right image as the background
    stitched = warped_right.copy()

    # Overlay the left image
    mask_left = warped_left > 0
    stitched[mask_left] = warped_left[mask_left]

    if blend:
        overlap = (warped_left > 0) & (warped_right > 0)
        stitched[overlap] = warped_left[overlap] // 2 + warped_right[overlap] // 2

    return stitched


if __name__ == "__main__":
    params = parse_args()
    outdir = params.output
    plt.rcParams["figure.figsize"] = [5, 5]

    # Read images
    right_origin, right_rgb, right_gray = read_image(params.right_img)
    left_origin, left_rgb, left_gray = read_image(params.left_img)

    # 1. Detect interest points and extract features
    kp_left, des_left = sift_point(left_gray)
    kp_right, des_right = sift_point(right_gray)

    # 2. Match features
    matches = matcher(kp_left, kp_right, des_left, des_right, 0.5)

    # 3. RANSAC to find homography matrix H
    inliers, H = ransac_homography(matches, threshold=0.3, iterations=2000)

    # 4. Warp image to create panoramic image
    panoramic = stitch_image(left_origin, right_origin, H, blend=True)

    # Save the panoramic image
    left_image_name = os.path.basename(params.left_img).split(".")[0]
    right_image_name = os.path.basename(params.right_img).split(".")[0]
    out_path = os.path.join(outdir, f"{left_image_name}_{right_image_name}_pano.jpg")
    cv2.imwrite(out_path, panoramic)
    print(f"Panoramic image saved to {out_path}")

    # Visualizations
    if params.visualize:
        right_kp_img = show_keypoints(right_gray, right_rgb, kp_right)
        left_kp_img = show_keypoints(left_gray, left_rgb, kp_left)

        img_concat = np.concatenate((left_kp_img, right_kp_img), axis=1)
        cv2.imwrite(os.path.join(outdir, "keypoints.jpg"), img_concat)
        show_matches(img_concat, matches, os.path.join(outdir, "matches.jpg"))
        show_matches(img_concat, matches[inliers], os.path.join(outdir, "inliers.jpg"))
