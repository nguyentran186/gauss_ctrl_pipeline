import os
import numpy as np
import open3d as o3d
from read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary

# Path to COLMAP output folder
colmap_sparse_path = "/workspace/data/sparse/0/"  # Adjust this to your path

# Read COLMAP files
cameras_file = os.path.join(colmap_sparse_path, "cameras.bin")
images_file = os.path.join(colmap_sparse_path, "images.bin")
points3D_file = os.path.join(colmap_sparse_path, "points3D.bin")

cameras = read_cameras_binary(cameras_file)
images = read_images_binary(images_file)
points3D = read_points3D_binary(points3D_file)

# Extract 3D points
points = np.array([point.xyz for point in points3D.values()])
colors = np.array([point.rgb for point in points3D.values()]) / 255.0  # Normalize RGB values

# Save the point cloud to a .ply file
output_ply_file = "/workspace/data/sparse/0/sparse_reconstruction.ply"  # Output file path
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Save the point cloud
o3d.io.write_point_cloud(output_ply_file, point_cloud)
print(f"Point cloud saved to {output_ply_file}")
