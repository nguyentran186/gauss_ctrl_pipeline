import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting


HARD_PROMPT = "fit to the background, do not add things"

def create_diffusion(pretrain_path="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                     torch_dtype=torch.float16,
                     variant="fp16",
                     device="cuda"
                     ):
    pipe = AutoPipelineForInpainting.from_pretrained(
        pretrain_path=pretrain_path,
        torch_dtype=torch_dtype,
        variant=variant).to(device)

    return pipe

def resize1024(image):
    return image.resize((1024, 1024), Image.LANCZOS)

def resize_and_save(images, original_size, save_paths):
    resized_images = []
    for image, save_path in zip(images, save_paths):
        resized_image = image.resize(original_size)
        resized_image.save(save_path)
        print(f"Saved image to {save_path}")
        resized_images.append(resized_image)
    return resized_images

def dilate_mask(mask_image, kernel_size=5, iterations=1):
    mask_image_np = np.array(mask_image)
    mask_gray = cv2.cvtColor(mask_image_np, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_dilated = cv2.dilate(mask_gray, kernel, iterations=iterations).clip(0, 255).astype(np.uint8)
    mask_dilated_pil = Image.fromarray(mask_dilated)

    return mask_dilated_pil