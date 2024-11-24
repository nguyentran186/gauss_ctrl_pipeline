import os
import cv2
import numpy as np
from skimage.exposure import match_histograms
from tqdm import tqdm  # For progress tracking

def histogram_matching_folders(folder1, folder2, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get lists of image files in both folders
    images1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Ensure both folders have the same number of images
    if len(images1) != len(images2):
        raise ValueError("The two folders must contain the same number of images.")

    print(f"Matching histograms for {len(images1)} image pairs...")

    # Process each image pair
    for img1_name, img2_name in tqdm(zip(images1, images2), total=len(images1)):
        # Read images
        img1_path = os.path.join(folder1, img1_name)
        img2_path = os.path.join(folder2, img2_name)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Convert images to RGB (OpenCV loads images as BGR)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Match histograms channel by channel
        matched_img = np.zeros_like(img1)
        for channel in range(3):  # RGB channels
            matched_img[..., channel] = match_histograms(img1[..., channel], img2[..., channel], multichannel=False)

        # Convert back to BGR for saving
        matched_img_bgr = cv2.cvtColor(matched_img, cv2.COLOR_RGB2BGR)
        
        # Save the matched image
        output_path = os.path.join(output_folder, img1_name)  # Save with the same name as the input
        cv2.imwrite(output_path, matched_img_bgr)

    print(f"Histogram matching completed. Results saved in '{output_folder}'.")

# Example usage
folder1 = "/workspace/knguyen/gaussian-splatting/data/statue/images"  # Folder with images to be mapped
folder2 = "/workspace/data/images"  # Folder with reference images
output_folder = "/workspace/knguyen/gaussian-splatting/data/statue/colored_images"  # Folder to save the mapped images

histogram_matching_folders(folder1, folder2, output_folder)
