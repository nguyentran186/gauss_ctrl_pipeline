# reference from https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers.utils import load_image
import os

def prepare_image(image):
    return image.resize((1024, 1024), Image.LANCZOS)

def resize_and_save(image, original_size, save_path):
    resized_image = image.resize(original_size)
    resized_image.save(save_path)
    return resized_image

def get_depth_map(image, depth_path='depth_map_new.png'):
    original_size = image.size
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    resize_and_save(image, original_size, depth_path)
    return image

if __name__ == '__main__':
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    
    data_path = '/home/ubuntu/workspace/bhrc/nam/gauss_ctrl_pipeline/data/statue_inpainted'
    output_path = './data/statue_inpainted'

    for i, file_path in enumerate(os.listdir(data_path)):
        if file_path=='depth_ori':
            continue    
        init_image = load_image(os.path.join(data_path, file_path)).convert("RGB")
        os.makedirs(os.path.join(output_path, 'depth_ori'), exist_ok=True)
        file_name = os.path.basename(file_path)
        depth_path = os.path.join(output_path, 'depth_ori', file_name)
        
        depth_image = get_depth_map(init_image, depth_path)
