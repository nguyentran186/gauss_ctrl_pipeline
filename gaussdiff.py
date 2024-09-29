import torch
import os
import helper
import gauss_ctrl_pipeline.utils as utils
from diffusers.utils import load_image
from helper import HARD_PROMPT


class GaussDiff:
    def __init__(self,
                 data_path: str,
                 output_path: str,
                 diffusion=helper.create_diffusion(),
                 device="cuda"
                 ):
        self.data_path = data_path
        self.images_path = os.path.join(data_path, 'images')
        self.masks_path = os.path.join(data_path, 'masks')
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.diffusion = diffusion
        self.device = device

        self.images = []
        self.masks = []
        self.sorted_files = sorted(os.listdir(self.images_path))
        self.prepare_data()

        self.generator = torch.Generator(device=device).manual_seed(0)
        
    def change_path(self,
                    data_path=None,
                    output_path=None
                    ):
        self.data_path = data_path if data_path else self.data_path
        self.images_path = os.path.join(data_path, 'images')
        self.masks_path = os.path.join(data_path, 'masks')
        self.output_path = output_path if output_path else self.output_path
        os.makedirs(self.output_path, exist_ok=True)

    def prepare_data(self, reset=True):
        if reset:
            self.images = []
            self.masks = []
        for file_name in self.sorted_files:
            image = load_image(os.path.join(self.images_path, file_name)).convert("RGB")
            image = helper.resize1024(image)
            mask = load_image(os.path.join(self.masks_path, file_name)).convert("RGB")
            mask = mask.point(lambda p: p * 255)
            mask = helper.dilate_mask(mask, kernel_size=20).convert('RGB')
            mask = helper.resize1024(mask)
            self.images.append(image)
            self.masks.append(mask)

    def edit_default(self, 
             ref_images_idx=None,
             prompt=HARD_PROMPT
             ):
        ref_images = [self.images[i] for i in ref_images_idx]
        ref_masks = [self.masks[i] for i in ref_images_idx]
        self.diffusion.unet.set_attn_processor(
            processor=utils.CrossViewAttnProcessor(
                self_attn_coeff=1,
                unet_chunk_size=2, 
                num_ref=len(ref_images))
        )
        for i in range(len(self.images)):
            images = [self.images[i]] + ref_images
            masks = [self.masks[i]] + ref_masks
            output = self.diffusion(
                [prompt] * len(images),
                image=images,
                mask_image=masks,
                num_inference_steps=20,
                generator=self.generator,
            ).images[0]
            output = output.resize(self.images[i].size)
            output.save(os.path.join(self.output_path, self.sorted_files[i]))