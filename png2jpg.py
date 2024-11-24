import os
from PIL import Image

# Specify the folder containing .png files
folder_path = "/workspace/knguyen/gaussian-splatting/data/statue/images"

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # Full path to the .png file
        png_path = os.path.join(folder_path, filename)
        
        # Create the new .jpg file path
        jpg_path = os.path.join(folder_path, filename.replace(".png", ".jpg"))
        
        # Open the .png image and convert to RGB
        with Image.open(png_path) as img:
            rgb_img = img.convert("RGB")  # Convert to RGB to avoid transparency issues
            rgb_img.save(jpg_path, "JPEG")  # Save as .jpg
        
        print(f"Converted: {filename} -> {os.path.basename(jpg_path)}")

print("Conversion complete!")
