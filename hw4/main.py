import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="3D Reconstruction")
    parser.add_argument(
        "--img1", type=str, default="data/tissue1.jpeg", help="Path to image 1"
    )
    parser.add_argument(
        "--img2", type=str, default="data/tissue2.jpeg", help="Path to image 2"
    )
    parser.add_argument(
        "--calib",
        type=str,
        default="data/my_calib.txt",
        help="Path to calibration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output files",
    )
    return parser.parse_args()


def sift_and_match(img1, img2):
    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Detect and compute keypoints and descriptors
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # Match features using KNN
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matching points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2


def normalize_points(points):
    mean = np.mean(points, axis=0)
    std = np.std(points)
    scale = np.sqrt(2) / std
    T = np.array(
        [[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1]]
    )
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    normalized_points = (T @ points_h.T).T

    return normalized_points[:, :2], T


def compute_fundamental_matrix(pts1, pts2):
    # Step 1: Normalize points
    pts1_normalized, T1 = normalize_points(pts1)
    pts2_normalized, T2 = normalize_points(pts2)

    # Step 2: Construct matrix A
    A = np.zeros((len(pts1), 9))
    for i, (p1, p2) in enumerate(zip(pts1_normalized, pts2_normalized)):
        x1, y1 = p1
        x2, y2 = p2
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

    # Step 3: Solve using SVD
    _, _, V = np.linalg.svd(A)
    F_normalized = V[-1].reshape((3, 3), order="F")

    # Step 4: Enforce rank-2 constraint
    U, S, V = np.linalg.svd(F_normalized)
    S[2] = 0
    F_normalized = U @ np.diag(S) @ V
    F_normalized = F_normalized / F_normalized[2, 2]

    # Step 5: Denormalize
    F = T2.T @ F_normalized @ T1
    return F / F[2, 2]


def compute_fundamental_matrix_ransac(pts1, pts2, threshold=1, max_iterations=2000):
    best_F = None
    max_inliers = 0
    best_inliers_mask = None

    for _ in range(max_iterations):
        # Randomly select 8 points
        idx = np.random.choice(len(pts1), 8, replace=False)
        subset_pts1 = pts1[idx]
        subset_pts2 = pts2[idx]

        F = compute_fundamental_matrix(subset_pts1, subset_pts2)

        # Calculate error and inliers
        def sampson_error(F, x1, x2):
            if x1.shape[1] == 2:
                x1 = np.hstack((x1, np.ones((x1.shape[0], 1))))
                x2 = np.hstack((x2, np.ones((x2.shape[0], 1))))

            F_src = F @ x1.T
            Ft_dst = F.T @ x2.T

            dst_F_src = np.sum(x2 * F_src.T, axis=1)

            return np.abs(dst_F_src) / np.sqrt(
                F_src[0] ** 2 + F_src[1] ** 2 + Ft_dst[0] ** 2 + Ft_dst[1] ** 2
            )

        errors = sampson_error(F, pts1, pts2)
        inliers_mask = errors < threshold
        num_inliers = np.sum(inliers_mask)

        # Update best F and inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = F
            best_inliers_mask = inliers_mask

    return best_F, best_inliers_mask


def compute_essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    S = (S[0] + S[1]) / 2  # Average the first two singular values
    E = U @ np.diag([S, S, 0]) @ Vt
    return E


def compute_camera_poses(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return R1, R2, t


def triangulate_points(pts1, pts2, P1, P2):
    """
    Triangulate 3D points given 2D correspondences and projection matrices.
    """
    if pts1.shape[1] == 2:
        pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    else:
        pts1_h = pts1
        pts2_h = pts2

    points_4d = []
    for i in range(pts1.shape[0]):
        A = np.zeros((4, 4))
        A[0] = pts1_h[i, 0] * P1[2] - P1[0]
        A[1] = pts1_h[i, 1] * P1[2] - P1[1]
        A[2] = pts2_h[i, 0] * P2[2] - P2[0]
        A[3] = pts2_h[i, 1] * P2[2] - P2[1]

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X /= X[3]  # Convert to non-homogeneous coordinates
        points_4d.append(X[:3])

    return np.array(points_4d)


def count_in_front_of_both_cameras(X, R1, t1, R2, t2):
    """
    Counts the number of 3D points in front of both cameras.

    Parameters:
        X (ndarray): 3D points of shape (N, 3).
        R1 (ndarray): Rotation matrix of camera1 (3, 3).
        t1 (ndarray): Translation vector of camera1 (3,).
        R2 (ndarray): Rotation matrix of camera2 (3, 3).
        t2 (ndarray): Translation vector of camera2 (3,).

    Returns:
        int: Number of points in front of both cameras.
    """
    # Compute camera centers
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2

    # Compute vectors to points for both cameras
    vectors_to_points_cam1 = X - C1
    vectors_to_points_cam2 = X - C2

    # Camera viewing directions (third row of rotation matrices)
    viewing_direction_cam1 = R1[2, :]
    viewing_direction_cam2 = R2[2, :]

    # Dot products to check if points are in front of each camera
    dot_products_cam1 = np.dot(vectors_to_points_cam1, viewing_direction_cam1)
    dot_products_cam2 = np.dot(vectors_to_points_cam2, viewing_direction_cam2)

    # Points are in front of both cameras if dot products > 0 for both
    in_front_both = (dot_products_cam1 > 0) & (dot_products_cam2 > 0)

    # Count points in front of both cameras
    in_front_count = np.sum(in_front_both)

    return in_front_count


def choose_best_camera_pose(R1, R2, t1, pts1, pts2, K1, K2):
    """
    Find the correct pose that satisfies the cheirality condition.
    """
    poses = [
        (R1, t1),
        (R1, -t1),
        (R2, t1),
        (R2, -t1),
    ]

    max_in_front = -1
    correct_idx = -1
    correct_pose = None
    correct_P1, correct_P2 = None, None

    for i, (R, t) in enumerate(poses):
        print(f"Pose {i + 1}")

        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R, t.reshape(-1, 1)))

        # Perform triangulation
        pts_3d = triangulate_points(pts1, pts2, P1, P2)

        # Check how many points are in front of both cameras
        in_front_count = count_in_front_of_both_cameras(
            pts_3d, np.eye(3), np.zeros(3), R, t
        )
        print(f"Points in front: {in_front_count}")

        if in_front_count > max_in_front:
            max_in_front = in_front_count
            correct_idx = i
            correct_pose = (R, t)
            correct_P1, correct_P2 = P1, P2
    print(f"Best pose: Pose {correct_idx + 1}")
    return (correct_pose, correct_P1, correct_P2, correct_idx)


def generate_obj_file(P, p_img2, M, tex_name, im_index, output_dir):
    """
    Converts 3D points and texture mapping information into an OBJ file and generates visualizations.

    Parameters:
        P (ndarray): 3D points (N x 3).
        p_img2 (ndarray): 2D texture coordinates (N x 2).
        M (ndarray): Visibility matrix (not fully implemented here).
        tex_name (str): Path to texture image.
        im_index (int): Model index for output filenames.
        output_dir (str): Directory to save output files.
    """
    # Load texture image and get size
    img = Image.open(tex_name)
    img_size = img.size  # (width, height)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------
    # Mesh triangulation
    # ----------------------------------------------------
    tri = Delaunay(p_img2)  # 2D Delaunay triangulation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(
        P[:, 0],
        P[:, 1],
        P[:, 2],
        triangles=tri.simplices,
        cmap="viridis",
        edgecolor="k",
    )
    plt.title("3D Mesh Triangulation")
    mesh_path = os.path.join(output_dir, f"mesh_triangulation_{im_index}.png")
    plt.savefig(mesh_path)
    plt.close(fig)

    # ----------------------------------------------------
    # Output .obj file
    # ----------------------------------------------------
    obj_filename = os.path.join(output_dir, f"model{im_index}.obj")
    mtl_filename = os.path.join(output_dir, f"model{im_index}.mtl")

    with open(obj_filename, "w") as obj_file:
        obj_file.write("# obj file\n")
        obj_file.write(f"mtllib model{im_index}.mtl\n\n")
        obj_file.write("usemtl Texture\n")

        # Write 3D vertex information
        for point in P:
            obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")

        # Write texture coordinates
        for coord in p_img2:
            u = coord[0] / img_size[0]
            v = 1 - (coord[1] / img_size[1])
            obj_file.write(f"vt {u} {v}\n")

        # Write face information
        for simplex in tri.simplices:
            # For simplicity, assuming all faces are visible
            obj_file.write(
                f"f {simplex[0]+1}/{simplex[0]+1} "
                f"{simplex[1]+1}/{simplex[1]+1} "
                f"{simplex[2]+1}/{simplex[2]+1}\n"
            )

    # ----------------------------------------------------
    # Output .mtl file
    # ----------------------------------------------------
    with open(mtl_filename, "w") as mtl_file:
        mtl_file.write("# MTL file\n")
        mtl_file.write("newmtl Texture\n")
        mtl_file.write("Ka 1 1 1\nKd 1 1 1\nKs 1 1 1\n")
        mtl_file.write(f"map_Kd {tex_name.replace('data', 'texture')}\n")

    print(f"Files saved: {obj_filename}, {mtl_filename}, and {mesh_path}")


def generate_mat_file(pts_3d, p_img2, M, tex_name, im_index, output_dir):
    import scipy.io

    # 保存 3D 點和 2D 投影點到 MATLAB 的 .mat 格式
    scipy.io.savemat(
        "mesh/matlab_input.mat",
        {
            "P": pts_3d,
            "p_img2": p_img2,
            "M": M,
            "tex_name": tex_name,
            "im_index": im_index,
            "output_dir": output_dir,
        },
    )

    print("Files saved: matlab_input.mat")


# ================== Plotting functions ==================


def plot_camera(R, t, ax, scale=0.1, depth=0.1, faceColor="grey"):
    C = -t  # camera center (in world coordinate system)

    # Generating camera coordinate axes
    axes = np.zeros((3, 6))
    axes[0, 1], axes[1, 3], axes[2, 5] = 1, 1, 1

    # Transforming to world coordinate system
    axes = R.T.dot(axes) + C[:, np.newaxis]

    # Plotting axes
    ax.plot3D(xs=axes[0, :2], ys=axes[1, :2], zs=axes[2, :2], c="r")
    ax.plot3D(xs=axes[0, 2:4], ys=axes[1, 2:4], zs=axes[2, 2:4], c="g")
    ax.plot3D(xs=axes[0, 4:], ys=axes[1, 4:], zs=axes[2, 4:], c="b")

    # generating 5 corners of camera polygon
    pt1 = np.array([[0, 0, 0]]).T  # camera centre
    pt2 = np.array([[scale, -scale, depth]]).T  # upper right
    pt3 = np.array([[scale, scale, depth]]).T  # lower right
    pt4 = np.array([[-scale, -scale, depth]]).T  # upper left
    pt5 = np.array([[-scale, scale, depth]]).T  # lower left
    pts = np.concatenate((pt1, pt2, pt3, pt4, pt5), axis=-1)

    # Transforming to world-coordinate system
    pts = R.T.dot(pts) + C[:, np.newaxis]
    ax.scatter3D(xs=pts[0, :], ys=pts[1, :], zs=pts[2, :], c="k")

    # Generating a list of vertices to be connected in polygon
    verts = [
        [pts[:, 0], pts[:, 1], pts[:, 2]],
        [pts[:, 0], pts[:, 2], pts[:, -1]],
        [pts[:, 0], pts[:, -1], pts[:, -2]],
        [pts[:, 0], pts[:, -2], pts[:, 1]],
    ]

    # Generating a polygon now..
    ax.add_collection3d(
        Poly3DCollection(
            verts, facecolors=faceColor, linewidths=1, edgecolors="k", alpha=0.25
        )
    )


def plot_epilines(img1, img2, lines, pts1, pts2, output_path=None):
    r, c = img1.shape
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(map(int, pt1)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(map(int, pt2)), 5, color, -1)

    if output_path:
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))

        axs[0].imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Epilines in Image 1")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Epilines in Image 2")
        axs[1].axis("off")

        plt.savefig(
            "output/epilines_combined_plot.jpg", bbox_inches="tight", pad_inches=0.1
        )

    return img1_color, img2_color


def plot_3d(
    pts_3d, R, t, title="3d points", ax=None, output_path=None, cameraConfig=None
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2])

    if cameraConfig:
        plot_camera(np.eye(3), np.zeros(3), ax, scale=0.1, depth=0.1, faceColor="blue")
        plot_camera(R, t, ax, **cameraConfig)

    if output_path:
        plt.savefig(output_path)


if __name__ == "__main__":

    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Read the images
    img1 = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(args.img2, cv2.IMREAD_GRAYSCALE)

    # Compute SIFT keypoints and descriptors
    pts1, pts2 = sift_and_match(img1, img2)

    # Compute the Fundamental Matrix
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    F, mask = compute_fundamental_matrix_ransac(
        pts1, pts2, threshold=1, max_iterations=100
    )

    # Filter points based on the mask
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Compute epilines for points in the first image and draw them on the second image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img1_epilines, img2_points = plot_epilines(
        img1, img2, lines1, pts1, pts2, output_path=os.path.join(output_dir, "epilines")
    )

    # Intrinsic matrix K
    K1 = np.loadtxt(args.calib, skiprows=3, max_rows=3)
    K2 = np.loadtxt(args.calib, skiprows=9, max_rows=3)

    # Compute Essential Matrix
    E = compute_essential_matrix(F, K1, K2)

    # Compute camera poses
    R1, R2, t = compute_camera_poses(E)

    # Plot 4 camera pose
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Pose")
    # Plot the first camera
    plot_camera(np.eye(3), np.zeros(3), ax, scale=0.1, depth=0.1, faceColor="blue")

    # Plot the second camera with four possible solutions
    plot_camera(R1, t, ax, scale=0.1, depth=0.1, faceColor="red")
    plot_camera(R1, -t, ax, scale=0.1, depth=0.1, faceColor="green")
    plot_camera(R2, t, ax, scale=0.1, depth=0.1, faceColor="yellow")
    plot_camera(R2, -t, ax, scale=0.1, depth=0.1, faceColor="purple")

    plt.savefig(os.path.join(output_dir, "camera_poses.png"))

    # Plot the 3D points with different camera poses
    fig = plt.figure(figsize=(15, 10))
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    poses = [
        (R1, t, "Pose 1"),
        (R1, -t, "Pose 2"),
        (R2, t, "Pose 3"),
        (R2, -t, "Pose 4"),
    ]

    for i, (R, t, title) in enumerate(poses, 1):
        P2 = K2 @ np.hstack((R, t.reshape(-1, 1)))
        pts_3d = triangulate_points(pts1, pts2, P1, P2)
        ax = fig.add_subplot(2, 2, i, projection="3d")
        cameraConfig = {"scale": 0.1, "depth": 0.1, "faceColor": "red"}
        plot_3d(pts_3d, R, t, title, ax, None, cameraConfig)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3dpts_with_camera_all.png"))

    # Choose the best camera pose
    correct_pose, P1, P2, idx = choose_best_camera_pose(R1, R2, t, pts1, pts2, K1, K2)
    pts_3d = triangulate_points(pts1, pts2, P1, P2)

    plot_3d(
        pts_3d,
        correct_pose[0],
        correct_pose[1],
        title=f"Best Camera Pose: Pose {idx + 1}",
        output_path=os.path.join(output_dir, "best_camera_pose.png"),
        cameraConfig={"scale": 0.1, "depth": 0.1, "faceColor": "red"},
    )

    generate_obj_file(pts_3d, pts1, P1, args.img1, 1, "mesh")
    generate_mat_file(pts_3d, pts2, P1, args.img1, 1, "mesh")

    # plt.show()
