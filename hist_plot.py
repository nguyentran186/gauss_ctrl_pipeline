import cv2
import matplotlib.pyplot as plt

def save_histograms(image1_path, image2_path, output_path):
    # Load the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Check if images are loaded
    if image1 is None or image2 is None:
        raise FileNotFoundError("One or both image paths are invalid.")
    
    # Convert images to RGB (OpenCV loads as BGR by default)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Colors for the histogram plots
    colors = ['Red', 'Green', 'Blue']
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    for i, color in enumerate(colors):
        plt.subplot(1, 3, i + 1)
        hist1 = cv2.calcHist([image1], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([image2], [i], None, [256], [0, 256])
        
        plt.plot(hist1, color=color.lower(), label=f"{color} - Image 1")
        plt.plot(hist2, color=color.lower(), linestyle="--", label=f"{color} - Image 2")
        
        plt.title(f"{color} Channel")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)  # Save the figure to a file
    print(f"Histogram saved to {output_path}")
    plt.close()  # Close the plot to free resources

# Paths to images and output file
image1_path = "/workspace/data/images/IMG_2727.png"
image2_path = "/workspace/knguyen/gaussian-splatting/data/statue/images/IMG_2727.jpg"
output_path = "/workspace/knguyen/gaussian-splatting/hist.png"

# Save the histograms
save_histograms(image1_path, image2_path, output_path)
