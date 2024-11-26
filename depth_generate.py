import os
import torch
import cv2
import numpy as np
import argparse
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def estimate_depth(image_path, model, transform):
    """
    Estimate the depth of an image using a preloaded MiDaS model and transform.

    Parameters:
        image_path (str): Path to the input image.
        model: Preloaded MiDaS model.
        transform: Preloaded transform for the MiDaS model.

    Returns:
        depth_map (numpy.ndarray): The estimated depth map.
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img)

    # Perform depth estimation
    with torch.no_grad():
        depth = model(input_batch)
        depth_map = depth.squeeze().cpu().numpy()

    # Normalize depth map for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)

    return depth_map_normalized


def process_images(input_folder, output_folder, model_type="DPT_Large"):
    """
    Process all images in a folder to estimate their depth maps and save them.

    Parameters:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save depth maps.
        model_type (str): Type of MiDaS model to use. Options are 'DPT_Large', 'DPT_Hybrid', etc.
    """
    # Load the model and transforms
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms")

    # Check model type for appropriate transform
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = transform.dpt_transform
    else:
        transform = transform.midas_transform

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"depth_{filename}")

            # Estimate depth
            depth_map = estimate_depth(input_path, model, transform)

            # Save the depth map
            cv2.imwrite(output_path, depth_map)
            print(f"Processed {filename} -> {output_path}")


def argument_parser():
    """
    Argument parser for the depth estimation script.

    Returns:
        argparse.ArgumentParser: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Depth estimation using MiDaS")

    parser.add_argument("--path", "-s", required=True, type=str,
                        help="Path to the folder containing input images.")
    parser.add_argument("--model-type", default="DPT_Large", type=str,
                        help="Type of MiDaS model to use. Options are 'DPT_Large', 'DPT_Hybrid', etc.")

    return parser


if __name__ == "__main__":
    # Parse arguments
    args = argument_parser().parse_args()
    input_folder = os.path.join(args.path, 'images')
    output_folder = os.path.join(args.path, 'depths')

    # Call the processing function with the provided arguments
    process_images(input_folder, output_folder, args.model_type)
