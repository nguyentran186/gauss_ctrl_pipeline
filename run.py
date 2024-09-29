import numpy as np
import cv2
import matplotlib.pyplot as plt

def project_points(image, K, R, T, depth_map):
    """
    Project 2D image onto a new view using intrinsic and extrinsic parameters.

    Parameters:
    image (ndarray): Input 2D image (H x W x C).
    K (ndarray): Intrinsic matrix (3 x 3).
    R (ndarray): Rotation matrix (3 x 3).
    T (ndarray): Translation vector (3 x 1).
    depth_map (ndarray): Depth map (H x W).

    Returns:
    projected_image (ndarray): Image rendered from the new view.
    """
    height, width = image.shape[:2]
    
    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert pixel coordinates to normalized image coordinates in source view
    x_normalized = (u - K[0, 2]) / K[0, 0]  # x coordinates normalized
    y_normalized = (v - K[1, 2]) / K[1, 1]  # y coordinates normalized

    # Construct the 3D point cloud from 2D image and depth
    X_3D = np.stack((x_normalized * depth_map, y_normalized * depth_map, depth_map), axis=-1)  # (H, W, 3)

    # Reshape to (N, 3) for matrix operations
    X_3D_flat = X_3D.reshape(-1, 3)  # Shape (H*W, 3)

    # Apply the extrinsic transformation
    X_3D_transformed = (R @ X_3D_flat.T + T.reshape(3, 1)).T  # Shape (H*W, 3)

    # Project the points onto the image plane
    X_projected = K @ X_3D_transformed.T  # Shape (3, H*W)

    # Normalize by the third coordinate (depth)
    points_2D = X_projected[:2, :] / X_projected[2, :]  # Shape (2, H*W)

    # Reshape back to image dimensions
    points_2D = points_2D.T  # Shape (H*W, 2)
    
    # Create an empty image for the projected view
    projected_image = np.zeros_like(image)

    # Map colors from the original image to the projected points
    for i in range(points_2D.shape[0]):
        x, y = int(points_2D[i, 0]), int(points_2D[i, 1])
        if 0 <= x < width and 0 <= y < height:
            projected_image[y, x] = image.flat[i * 3:(i * 3) + 3]  # Assign color

    return projected_image

def compute_R_ts_T_ts(c2w_s, c2w_t):
    # Extract rotation (upper-left 3x3) and translation (rightmost column) from source and target c2w matrices
    R_s = c2w_s[:3, :3]  # Source rotation
    T_s = c2w_s[:3, 3]   # Source translation
    
    R_t = c2w_t[:3, :3]  # Target rotation
    T_t = c2w_t[:3, 3]   # Target translation

    # Compute relative rotation: R_ts = R_t^T * R_s
    R_ts = np.dot(R_t.T, R_s)

    # Compute relative translation: T_ts = R_t^T * (T_s - T_t)
    T_ts = np.dot(R_t.T, (T_s - T_t))

    return R_ts, T_ts

# Load intrinsic and extrinsic matrices
K_s = np.load('/home/nguyen/.code/GS-HCMUT/gaussian_splatting/output/statue/train/ours_30000/intri/IMG_2707.npy')
K_t = np.load('/home/nguyen/.code/GS-HCMUT/gaussian_splatting/output/statue/train/ours_30000/intri/IMG_2714.npy')
c2w_s = np.load('/home/nguyen/.code/GS-HCMUT/gaussian_splatting/output/statue/train/ours_30000/c2w/IMG_2707.npy')
c2w_t = np.load('/home/nguyen/.code/GS-HCMUT/gaussian_splatting/output/statue/train/ours_30000/c2w/IMG_2714.npy')

# Compute the relative rotation and translation
R_ts, T_ts = compute_R_ts_T_ts(c2w_s, c2w_t)

# Load RGB image and depth map
rgb_image = cv2.imread('/home/nguyen/.code/GS-HCMUT/gaussian_splatting/data/statue/images/IMG_2707.jpg')
depth_map = np.load('/home/nguyen/.code/GS-HCMUT/gaussian_splatting/output/statue/train/ours_30000/depth/IMG_2707.npy')

projected_image = project_points(rgb_image, K_s, R_ts, T_ts, depth_map)
breakpoint()
cv2.imwrite('test.png', projected_image)