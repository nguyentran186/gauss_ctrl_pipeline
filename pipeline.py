# reference from https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils import load_image
from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
import os
import cv2
import utils_nam as utils

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.unet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0.4,
                        unet_chunk_size=1))
pipe.controlnet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0,
                        unet_chunk_size=1)) 

def prepare_image(image):
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
    """
    Dilates the input mask image.
    
    Parameters:
        mask_image (PIL.Image): The input mask as a PIL image.
        kernel_size (int): Size of the kernel to use for dilation (default is 5x5).
        iterations (int): Number of dilation iterations to apply (default is 1).
    
    Returns:
        PIL.Image: The dilated mask as a PIL image.
    """
    # Convert the PIL image to a NumPy array
    mask_image_np = np.array(mask_image)

    # Convert the mask to grayscale (if it is not already)
    mask_gray = cv2.cvtColor(mask_image_np, cv2.COLOR_RGB2GRAY)

    # Define a kernel for dilation (kernel_size x kernel_size)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation
    mask_dilated = cv2.dilate(mask_gray, kernel, iterations=iterations).clip(0, 255).astype(np.uint8)

    # Convert the dilated mask back to a PIL image
    mask_dilated_pil = Image.fromarray(mask_dilated)

    return mask_dilated_pil

if __name__ == '__main__':
    prompt = "fit to surrounding background and do not add anything into image"
    images_path = '/home/ubuntu/workspace/bhrc/nam/gaussctrl/data/data/statue/images'
    inpainted_depth_path = './data/statue_inpainted/depth'
    output_path = './data/inpainted/'
    os.makedirs(output_path, exist_ok=True)
    init_images = []
    depth_images = []
    mask_images = []
    for file_name in os.listdir(images_path):
        print('file_name: ', file_name)
        init_image = load_image(os.path.join(images_path, file_name)).convert("RGB")
        mask_image = load_image(os.path.join(images_path.split('images')[0],
                                             'mask',
                                             file_name[:-len('jpg')]+'png')).convert("RGB")
        
        mask_image = mask_image.point(lambda p: p * 255)
        mask_image = dilate_mask(mask_image, kernel_size=15).convert('RGB')

        controlnet_conditioning_scale = 0.5  # recommended for good generalization
        depth_image = load_image(os.path.join(inpainted_depth_path,
                                              file_name[:-len('jpg')]+'png')).convert("RGB")
        init_images.append(prepare_image(init_image))
        depth_images.append(prepare_image(depth_image))
        mask_images.append(prepare_image(mask_image))
    generator = torch.Generator(device="cuda").manual_seed(0)
    
    for i in range(len(init_images)):
        inpaint_idx = i
        init_img = [init_images[inpaint_idx]] + init_images[:inpaint_idx] + init_images[inpaint_idx + 1:]
        mask_img = [mask_images[inpaint_idx]] + mask_images[:inpaint_idx] + mask_images[inpaint_idx + 1:]
        depth_img = [depth_images[inpaint_idx]] + depth_images[:inpaint_idx] + depth_images[inpaint_idx + 1:]
        images = pipe(
            [prompt] * len(init_images),
            image=init_img,
            control_image=depth_img,
            mask_image=mask_img,
            num_inference_steps=20,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images[0]
        init_images[inpaint_idx] = images
        resize_and_save([images], init_image.size, [os.path.join(output_path, os.listdir(images_path)[i])])