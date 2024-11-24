from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import os
from PIL import Image
from visualize.cam_extract import cam_distances
import utils as utils
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Inpainting with Stable Diffusion XL")
    
    parser.add_argument("--cross_att", action="store_true", help="Enable cross attention")
    
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder")
    
    parser.add_argument("--prompt", type=str, default="""Restore the structure and consistency of the image.\n
                        Fill in the broken or distorted areas with natural, seamless transitions,\n
                        maintaining the original details and proportions. Ensure that the perspective\n
                        is accurate and that the texture and color transitions are smooth. Repair any missing\n
                        or misaligned elements, ensuring the scene looks cohesive and realistic, as if it was never\n
                        altered.""", help="Prompt for inpainting")
    
    parser.add_argument("--data_folder", type=str, default="/workspace/data", help="Data folder")
    
    parser.add_argument("--anchor", type=str, default="IMG_2707.jpg", help="Anchor image for cross attention")\
    
    parser.add_argument("--strength", type=float, default=0.9, help="Strength of the inpainting")
    
    parser.add_argument("--sparse_path", type=str, default="/workspace/data/sparse/0/", help="Path to COLMAP sparse folder")
    
    parser.add_argument("--cross_eff", type=float, default=0.2, help="Cross attention coefficent")
    
    return parser.parse_args()


def prepare_pipe(args):
    # Initialize the pipeline
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    if args.cross_att:
        pipe.unet.set_attn_processor(
                            processor=utils.CrossViewAttnProcessor(self_attn_coeff=args.cross_eff,
                            unet_chunk_size=2, num_ref=2))
    return pipe

def generate(images_folder, masks_folder, output_folder, prompt, pipe, args):
    generator = torch.Generator(device="cuda").manual_seed(0)

    distance = cam_distances(colmap_sparse_path = args.sparse_path, input_image = args.anchor, target_range=(0,1))
    
    if args.cross_att:
        original_image = load_image(os.path.join(images_folder, args.anchor[:-len('jpg')] + 'png'))
        image = original_image.resize((1024, 1024))  # Resize to 1024x1024 for processing
        mask_image = load_image(os.path.join(masks_folder, args.anchor))
        mask_image = mask_image.resize((1024, 1024))
        ref_image = image
        ref_mask = mask_image

    # Process each image-mask pair
    for file_name in os.listdir(images_folder):
        if file_name.endswith('png'):
            image_path = os.path.join(images_folder, file_name)
            mask_path = os.path.join(masks_folder, file_name[:-len('png')] + 'jpg')

            # Ensure the corresponding mask exists
            if not os.path.exists(mask_path):
                print(f"Mask for {file_name} not found, skipping...")
                continue

            # Load image and mask
            original_image = load_image(image_path)  # Load original image for size reference
            original_size = original_image.size  # (width, height)
            image = original_image.resize((1024, 1024))  # Resize to 1024x1024 for processing
            mask_image = load_image(mask_path)
            mask_image = mask_image.resize((1024, 1024))

            # Perform inpainting
            print(f"Processing: {file_name}")
            if not distance.get(file_name[:-len('png')] + 'jpg', False):
                continue
            if args.cross_att:
                result = pipe(
                    prompt=[prompt]*2,
                    image=[image, image],
                    mask_image=[mask_image, mask_image],
                    guidance_scale=8.0,
                    num_inference_steps=20,
                    strength=0.9,
                    generator=generator,
                ).images[0]
                # result = pipe(
                #     prompt=[prompt]*2,
                #     image=[image, ref_image],
                #     mask_image=[mask_image, ref_mask],
                #     guidance_scale=8.0,
                #     num_inference_steps=20,
                #     strength=0.9,
                #     generator=generator,
                # ).images[0]
            else:
                result = pipe(
                    prompt=prompt,
                    image=image,
                    mask_image=mask_image,
                    guidance_scale=8.0,
                    num_inference_steps=20,
                    strength=args.strength,
                    generator=generator,
                ).images[0]

            # Resize the result back to the original size
            result = result.resize(original_size)

            # Save the result
            output_path = os.path.join(output_folder, 'images', file_name.replace('png', 'jpg'))
            result.save(output_path)
            print(f"Saved: {output_path}")
            
def main(args):
    pipe = prepare_pipe(args)
    # Define input directories and output directory
    data_folder = args.data_folder
    images_folder = os.path.join(data_folder, "images")
    masks_folder = os.path.join(data_folder, "seg")
    output_folder = args.output_folder
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(output_folder, current_time)
    prompt = args.prompt
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    
    # Save the prompt and args in the output folder as a text file
    prompt_path = os.path.join(output_folder, "config.txt")
    with open(prompt_path, "w") as f:
        f.write("Arguments:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    generate(images_folder, masks_folder, output_folder, prompt, pipe, args)
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
