import os
import numpy as np
from visualize.read_write_model import read_images_binary
import math
import json
def extract_cams(colmap_sparse_path):
    # Path to the COLMAP sparse folder
    images_file = os.path.join(colmap_sparse_path, "images.bin")

    # Read the images.bin file
    images = read_images_binary(images_file)

    # Extract camera positions
    camera_positions = {}
    for image_id, image in images.items():
        # Camera position (C) is calculated as: C = -R.T @ t
        rotation_matrix = image.qvec2rotmat()
        translation_vector = image.tvec
        camera_position = -np.dot(rotation_matrix.T, translation_vector)
        camera_positions[image.name] = camera_position
    return camera_positions

def calculate_distances(image_name, camera_positions):
    """
    Calculate distances from the input image to all other images.

    Args:
        image_name (str): The name of the input image.
        camera_positions (dict): A dictionary of image names and their 3D positions.

    Returns:
        dict: A dictionary with other image names as keys and their distances as values.
    """
    if image_name not in camera_positions:
        raise ValueError(f"Image '{image_name}' not found in camera positions.")
    
    input_position = camera_positions[image_name]
    distances = {}
    for name, position in camera_positions.items():
        if name != image_name:
            distance = np.linalg.norm(input_position - position)
            distances[name] = distance
    return distances

def normalize_distances(distances, target_range=(0, 1)):
    """
    Normalize the distances to a given range.

    Args:
        distances (dict): A dictionary where keys are image names and values are distances.
        target_range (tuple): The target range to normalize to (min, max).

    Returns:
        dict: A dictionary with image names as keys and normalized distances as values.
    """
    min_dist = 0
    max_dist = max(distances.values())
    min_target, max_target = target_range

    normalized_distances = {}
    for name, dist in distances.items():
        # Apply normalization formula
        normalized_value = (
            (dist - min_dist) / (max_dist - min_dist) * (max_target - min_target) + min_target
        )
        normalized_distances[name] = max(0.2,min(0.4, 1/(1+math.exp(-10*normalized_value+5))))

    return normalized_distances

def cam_distances(colmap_sparse_path = "/workspace/data/sparse/0/", input_image = "IMG_2707.jpg", target_range=(0.5, 0.99)):
    camera_positions = extract_cams(colmap_sparse_path)
    # file_path = "/workspace/knguyen/sdxl/visualize/position.json"
    # with open(file_path, "r") as file:
    #     camera_positions = json.load(file)
    # camera_positions = {key: np.array(value) for key, value in camera_positions.items()}
    # Example usage
    distances = calculate_distances(input_image, camera_positions)
    # Open and read the JSON file
    for img_name, dist in distances.items():
        print(f"  To {img_name}: {dist:.2f}")
    distances = normalize_distances(distances, target_range)
    return distances
    

if __name__ == "__main__":
    colmap_sparse_path = "/workspace/data/sparse/0/"  # Adjust to your path
    input_image = "IMG_2707.jpg"
    target_range=(0, 1)
    distances = cam_distances(colmap_sparse_path, input_image, target_range)
    # Print the distances
    print(f"Distances from '{input_image}':")
    for img_name, dist in distances.items():
        print(f"  To {img_name}: {dist:.2f}")
    
