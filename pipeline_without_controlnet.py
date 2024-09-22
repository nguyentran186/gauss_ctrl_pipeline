# reference from https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
import torch
import numpy as np
from PIL import Image
from diffusers.utils import load_image
import os
import cv2
import utils_nam as utils
from diffusers import AutoPipelineForInpainting
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

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
    prompt = "fit to the background, do not add things"
    # negative_prompt = 'person, human, baby, object, bucket, animal'
    images_path = '/home/ubuntu/workspace/bhrc/nam/gaussctrl/data/data/statue/images'
    inpainted_depth_path = './data/statue_inpainted/depth'
    output_path = './data/inpainted/'
    os.makedirs(output_path, exist_ok=True)
    init_images = []
    depth_images = []
    mask_images = []
    files_list = sorted(os.listdir(images_path))
    for file_name in files_list:
        print('file_name: ', file_name)
        init_image = load_image(os.path.join(images_path, file_name)).convert("RGB")
        mask_image = load_image(os.path.join(images_path.split('images')[0],
                                             'mask',
                                             file_name[:-len('jpg')]+'png')).convert("RGB")
        
        mask_image = mask_image.point(lambda p: p * 255)
        mask_image = dilate_mask(mask_image, kernel_size=20).convert('RGB')

        controlnet_conditioning_scale = 0.5  # recommended for good generalization
        depth_image = load_image(os.path.join(inpainted_depth_path,
                                              file_name[:-len('jpg')]+'png')).convert("RGB")
        init_images.append(prepare_image(init_image))
        depth_images.append(prepare_image(depth_image))
        mask_images.append(prepare_image(mask_image))
    generator = torch.Generator(device="cuda").manual_seed(0)

    ref_images = []
    idx = []
    # nguyen ma giao
    magiao_idx = 2
    init_images = [init_images[magiao_idx]] + init_images[:magiao_idx] + init_images[magiao_idx + 1:]
    mask_images = [mask_images[magiao_idx]] + mask_images[:magiao_idx] + mask_images[magiao_idx + 1:]
    depth_images = [depth_images[magiao_idx]] + depth_images[:magiao_idx] + depth_images[magiao_idx + 1:]
    files_list = [files_list[magiao_idx]] + files_list[:magiao_idx] + files_list[magiao_idx + 1:]
    
    for i in range(len(init_images)):
        inpaint_idx = i
        # init_img = [init_images[inpaint_idx]] + init_images[:inpaint_idx] + init_images[inpaint_idx + 1:]
        # mask_img = [mask_images[inpaint_idx]] + mask_images[:inpaint_idx] + mask_images[inpaint_idx + 1:]
        # depth_img = [depth_images[inpaint_idx]] + depth_images[:inpaint_idx] + depth_images[inpaint_idx + 1:]
        init_img = [init_images[inpaint_idx]] + ref_images
        mask_img = [mask_images[inpaint_idx]] + [mask_images[id] for id in idx]
        depth_img = [depth_images[inpaint_idx]] + [depth_images[id] for id in idx]
        pipe.unet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=1,
                        unet_chunk_size=2, num_ref=len(init_img)))
        images = pipe(
            [prompt] * len(init_img),
            # [negative_prompt] * len(init_img),
            image=init_img,
            control_image=depth_img,
            mask_image=mask_img,
            num_inference_steps=20,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images[0]
        ref_images.append(images)
        idx.append(inpaint_idx)
        resize_and_save([images], init_image.size, [os.path.join(output_path, files_list[i])])