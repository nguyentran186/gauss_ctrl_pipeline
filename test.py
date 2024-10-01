import numpy as np
import json

# Example sorted list of image file names (these should be sorted according to your criteria)
sorted_image_list = [
    "images/frame_00096.jpg", 
    "images/frame_00095.jpg", 
    "images/frame_00094.jpg"
    # Add all other image files here in sorted order
]

# Load the JSON file with the transformation data
with open('/home/ubuntu/workspace/bhrc/nam/gauss_ctrl_pipeline/data/bear/transforms.json', 'r') as f:
    data = json.load(f)

# Function to extract position and orientation from the transformation matrix
def extract_position_orientation(transform_matrix):
    # Convert the matrix to a NumPy array for easy manipulation
    matrix = np.array(transform_matrix)

    # Extract the translation vector (position)
    position = matrix[:3, 3]

    # Extract the rotation matrix (orientation)
    rotation_matrix = matrix[:3, :3]

    return position, rotation_matrix

# Initialize empty lists to store positions and orientations
positions = []
orientations = []

# Iterate over the sorted image list
for image_file in sorted_image_list:
    # Find the corresponding frame in the JSON data
    for frame in data['frames']:
        if frame['file_path'] == image_file:
            # Extract position and orientation from the transformation matrix
            position, orientation = extract_position_orientation(frame['transform_matrix'])
            
            # Append the position and orientation to the respective lists
            positions.append(position)
            orientations.append(orientation)
            break

# Output the positions and orientations
print("Positions:", positions)
print("Orientations (Rotation Matrices):", orientations)
