import torch
import os
import helper
import utils
import json
import numpy as np
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
        self.transform_path = os.path.join(data_path, 'transforms.json')
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.diffusion = diffusion
        self.device = device

        self.largest_mask = {
            'id': 0,
            'area': 0
        }
        self.images = []
        self.masks = []
        self.positions = []
        self.orientations = []
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
    
    def extract_position_orientation(self, transform_matrix):
        matrix = np.array(transform_matrix)
        position = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]

        return position, rotation_matrix

    def prepare_data(self, reset=True):
        if reset:
            self.images = []
            self.masks = []

        with open(self.transform_path, 'r') as f:
            data = json.load(f)
            
        for file_name_id in range(len(self.sorted_files)):
            file_name = self.sorted_files[file_name_id]
            image = load_image(os.path.join(self.images_path, file_name)).convert("RGB")
            image = helper.resize1024(image)
            self.images.append(image)

            mask = load_image(os.path.join(self.masks_path, file_name)).convert("RGB")
            mask = mask.point(lambda p: p * 255)
            mask = helper.dilate_mask(mask, kernel_size=20).convert('RGB')
            mask = helper.resize1024(mask)
            self.masks.append(mask)
            
            area = np.sum(np.array(mask) > 0)
            if area > self.largest_mask['area']:
                self.largest_mask['id'] = file_name_id
                self.largest_mask['area'] = area

            file_path = 'images/'+file_name
            for frame in data['frames']:
                if frame['file_path'] == file_path:
                    position, orientation = self.extract_position_orientation(frame['transform_matrix'])
                    self.positions.append(position)
                    self.orientations.append(orientation)
                    break

    def calculate_distance(self, i, j):
        return np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[j]))

    def get_anchor_id(self):
        return self.largest_mask['id']

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
                num_ref=len(ref_images)+1)
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


if __name__ == "__main__":
    data_path = '/home/ubuntu/workspace/bhrc/nam/gauss_ctrl_pipeline/data/bear'
    output_path = '/home/ubuntu/workspace/bhrc/nam/gauss_ctrl_pipeline/output/bear'
    ref_images_idx = [0, 1, 2]
    prompt = HARD_PROMPT
    gauss_diff = GaussDiff(data_path, output_path)
    # gauss_diff.edit_default(ref_images_idx, prompt)