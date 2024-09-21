from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import os


if __name__ == "__main__":
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
    prompt = "fit to surrounding background and do not add anything into image"
    data_path = '/home/ubuntu/workspace/bhrc/nam/gaussctrl/data/data/statue/images'
    os.makedirs('./data/inpainted', exist_ok=True)
    for i, file_name in enumerate(os.listdir(data_path)):
        image = load_image(os.path.join(data_path, file_name)).convert("RGB").resize((1024, 1024))
        mask_image = load_image(os.path.join(data_path.split('images')[0], 'mask', file_name[:-len('jpg')]+'png')).convert("RGB").resize((1024, 1024))
        generator = torch.Generator(device="cuda").manual_seed(0)
        image_edited = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=8.0,
            num_inference_steps=20,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=generator,
            ).images[0]
        image_edited.save(f"./data/inpainted/inpainting_{i}.png")