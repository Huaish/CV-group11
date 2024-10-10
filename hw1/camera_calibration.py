import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from scipy.spatial.transform import Rotation as R
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = img[0].shape
rvecs = []
tvecs = []
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
# Vr = np.array(rvecs)
# Tr = np.array(tvecs)
# extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
"""
Write your code here
"""

def compute_homography(objpoints, imgpoints):
    # Compute homography matrix
    # objpoints are points in 3D space, imgpoints are 2D points in the image
    A = []

    # Convert object points and image points to 2D
    for i in range(len(objpoints)):
        X, Y = objpoints[i][0], objpoints[i][1]
        u, v = imgpoints[i][0][0], imgpoints[i][0][1]
        
        # Two corresponding points generate two equations
        A.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
    
    # Convert A to matrix
    A = np.array(A)
    
    # Use SVD to solve for A * h = 0
    U, S, Vh = np.linalg.svd(A)
    
    # The last row of Vh is the solution corresponding to the smallest singular value, which is the h we require
    H = Vh[-1].reshape(3, 3)
    
    # Normalize H so that H[2,2] = 1
    H = H / H[-1, -1]
    return H

def vij(H, i, j):
    # Calculate Vij for intrinsics parameter calculations
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])

# Calculate the intrinsics matrix K
def compute_intrinsics(H_list):
    V = []
    for H in H_list:
        V.append(vij(H, 0, 1))
        V.append(vij(H, 0, 0) - vij(H, 1, 1))
    
    V = np.array(V)
    _, _, Vh = np.linalg.svd(V)
    b = Vh[-1]
    B11, B12, B22, B13, B23, B33 = b
    
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])

    """
    method 1: Cholesky decomposition
    """
    # try:
    #     L = np.linalg.cholesky(B)
    # except np. linalg.LinAlgError as e:
    #     print('B is not positive definite. \n'
    #     'Try -B.')
    #     L = np. linalg.cholesky(-B)
    
    # K = L[2, 2] * np.linalg.inv(L).T

    """
    method 2: Intrisic Parameters
    """
    cy = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lamda = B33 - (B13**2 + cy*(B12*B13 - B11*B23)) / B11
    fx = np.sqrt(lamda / B11)
    fy = np.sqrt(lamda * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * fx**2 * fy / lamda
    cx = (gamma * cy / fy) - (B13 * fx**2 / lamda)
    
    # Combined intrinsic matrix K
    K = np.array([[fx, gamma, cx],
                  [0,  fy,    cy],
                  [0,  0,     1]])
    
    return K


def compute_extrinsics(K, H):
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    lambda_ = 1/np.linalg.norm(np.dot(np.linalg.inv(K), h1))

    r1 = lambda_*np.dot(np.linalg.inv(K), h1)
    r2 = lambda_*np.dot(np.linalg.inv(K), h2)
    r3 = np.cross(r1, r2)
    r_matrix = np.array([r1, r2, r3]).T

    t = lambda_*np.dot(np.linalg.inv(K), h3)
    r = R.from_matrix(r_matrix).as_rotvec()

    rvecs.append(r)
    tvecs.append(t)

"""
Main function part: Call the above functions step by step to complete camera correction
"""
H_matrices = [compute_homography(objp, imgp) for objp, imgp in zip(objpoints, imgpoints)]
K = compute_intrinsics(H_matrices)

for H in H_matrices:
    compute_extrinsics(K, H)

Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)

# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection = '3d')
# ax = fig.gca(projection='3d')
# camera setting
camera_matrix = K
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards

min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()

#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
