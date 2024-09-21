import torch, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from typing import Optional, Type, List
from typing_extensions import Literal
from dataclasses import dataclass, field


# from lang_sam import LangSAM
import utils
from utils import saving_image
from copy import deepcopy
from rich.progress import Console

import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from diffusers.models.attention_processor import AttnProcessor
from ControlNetDatamanager import DataManagerConfig, DataManager
import cv2

CONSOLE = Console(width=120)

@dataclass
class GaussCtrlPipelineConfig:
    datamanager: DataManagerConfig = DataManagerConfig()
    render_rate: int = 500
    prompt: str = "fit with the background, do not add new things"
    guidance_scale: float = 5
    num_inference_steps: int = 20
    chunk_size: int = 1
    ref_view_num: int = 1
    diffusion_ckpt: str = 'CompVis/stable-diffusion-v1-4'
        

class GaussCtrlPipeline:
    config: GaussCtrlPipelineConfig
    def __init__(
        self,
        config: GaussCtrlPipelineConfig,
        images_path,
        ):
        self.datamanager: DataManager = DataManager(config.datamanager, images_path)
        # self.datamanager.to(device)
        
        # self.langsam = LangSAM()
        self.images_path = images_path
        
        self.pipe_device = 'cuda:0'
        self.config = config
        
        self.ddim_scheduler = DDIMScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        # controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=torch.float16)
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(self.config.diffusion_ckpt, controlnet=controlnet).to(self.pipe_device).to(torch.float16)
        self.pipe.to(self.pipe_device)
    
        self.prompt = self.config.prompt
        added_prompt = 'best quality, extremely detailed'
        self.positive_prompt = self.prompt + ', ' + added_prompt
        self.negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    
        random.seed(13789)
        # view_num = len(self.datamanager.train_images)
        view_num = 5
        anchors = [(view_num * i) // self.config.ref_view_num for i in range(self.config.ref_view_num)] + [view_num]
        self.ref_indices = [random.randint(anchor, anchors[idx+1]) for idx, anchor in enumerate(anchors[:-1])]
        self.num_ref_views = len(self.ref_indices)
        
        self.num_inference_steps = self.config.num_inference_steps
        self.guidance_scale = self.config.guidance_scale
        self.controlnet_conditioning_scale = 1.0
        self.eta = 0.0
        self.chunk_size = self.config.chunk_size

    def dilate_image(self, masked_image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(masked_image, kernel, iterations=1)
        # save the output image
        cv2.imwrite('dilated_image.jpg', dilated_image*255)
        return dilated_image
        
        
    def render_reverse(self, has_depth=False):
        # Implement the rendering logic
        for index in range(len(self.datamanager.train_images)):
            data = self.datamanager.train_images[index]
            size = (512, 512)
            # breakpoint()
            pil_image = Image.fromarray(data["image"].astype(np.uint8))
            data['image'] = (cv2.resize(data['image'], size) / 255.0).astype(np.float32)
            data['depth_image'] = np.expand_dims(cv2.resize(data['depth_image'], size), axis=-1)
            data['mask_image'] = self.dilate_image(data['mask_image'], kernel_size=15)
            data['mask_image'] = cv2.resize(data['mask_image'], size).astype(np.float32)
            
            rendered_rgb = torch.from_numpy(data['image']).to(torch.float16).to(self.pipe_device)
            rendered_depth = torch.from_numpy(data['depth_image']).to(torch.float16).to(self.pipe_device)
        
            self.pipe.unet.set_attn_processor(processor=AttnProcessor())
            self.pipe.controlnet.set_attn_processor(processor=AttnProcessor()) 
            init_latent = self.image2latent(rendered_rgb)
            disparity = self.depth2disparity_torch(rendered_depth[:,:,0][None]) 
            
            self.pipe.scheduler = self.ddim_inverser
            latent, _ = self.pipe(prompt=self.positive_prompt, #  placeholder here, since cfg=0
                                num_inference_steps=self.num_inference_steps, 
                                latents=init_latent, 
                                image=disparity, return_dict=False, guidance_scale=0, output_type='latent')
            
            latent = latent.cpu()
            # save_path = os.path.join(self.images_path, 'latent', f'{name}.pt')
            # torch.save(latent, save_path)
            self.update_datasets(index, rendered_rgb.cpu(), rendered_depth, latent, None)
            
    def edit_images(self):
        # Implement the image editing logic
        self.pipe.scheduler = self.ddim_scheduler
        self.pipe.unet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0.6,
                        unet_chunk_size=2))
        self.pipe.controlnet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0,
                        unet_chunk_size=2)) 
        
        CONSOLE.print("Done Resetting Attention Processor", style="bold blue")
        
        print("#############################")
        CONSOLE.print("Start Editing: ", style="bold yellow")
        CONSOLE.print(f"Reference views are {[j+1 for j in self.ref_indices]}", style="bold yellow")
        print("#############################")
        ref_disparity_list = []
        ref_z0_list = []
        for ref_idx in self.ref_indices:
            ref_data = deepcopy(self.datamanager.train_data[ref_idx])
            ref_disparity = self.depth2disparity(ref_data['depth_image']) 
            ref_z0 = ref_data['z_0_image']
            ref_disparity_list.append(ref_disparity)
            ref_z0_list.append(ref_z0) 

        ref_disparities = np.concatenate(ref_disparity_list, axis=0)
        ref_z0s = np.concatenate(ref_z0_list, axis=0)
        ref_disparity_torch = torch.from_numpy(ref_disparities.copy()).to(torch.float16).to(self.pipe_device)
        ref_z0_torch = torch.from_numpy(ref_z0s.copy()).to(torch.float16).to(self.pipe_device)

        # Edit images in chunk
        for idx in range(0, len(self.datamanager.train_data), self.chunk_size): 
            chunked_data = self.datamanager.train_data[idx: idx+self.chunk_size]
            
            indices = [current_data['image_idx'] for current_data in chunked_data]
            mask_images = [current_data['mask_image'] for current_data in chunked_data if 'mask_image' in current_data.keys()] 
            unedited_images = [current_data['unedited_image'] for current_data in chunked_data]
            CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

            depths = [self.depth2disparity(current_data['depth_image']) for current_data in chunked_data]
            disparities = np.concatenate(depths, axis=0)
            disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(self.pipe_device)

            z_0_images = [current_data['z_0_image'] for current_data in chunked_data] # list of np array
            z0s = np.concatenate(z_0_images, axis=0)
            latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16).to(self.pipe_device)

            disp_ctrl_chunk = torch.concatenate((ref_disparity_torch, disparities_torch), dim=0)
            latents_chunk = torch.concatenate((ref_z0_torch, latents_torch), dim=0)
            generator = torch.Generator(device="cuda").manual_seed(0)
            chunk_edited = self.pipe(
                                prompt=[self.positive_prompt] * (self.num_ref_views+len(chunked_data)),
                                negative_prompt=[self.negative_prompts] * (self.num_ref_views+len(chunked_data)),
                                latents=latents_chunk,
                                image=disp_ctrl_chunk,
                                num_inference_steps=self.num_inference_steps,
                                guidance_scale=self.guidance_scale,
                                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                                eta=self.eta,
                                output_type='pt',
                                generator=generator,
                            ).images[self.num_ref_views:]
            chunk_edited = chunk_edited.cpu() 

            # Insert edited images back to train data for training
            for local_idx, edited_image in enumerate(chunk_edited):
                global_idx = indices[local_idx]

                bg_cntrl_edited_image = edited_image
                if mask_images != []:
                    mask = torch.from_numpy(mask_images[local_idx])
                    bg_mask = 1 - mask

                    unedited_image = unedited_images[local_idx].permute(2,0,1)
                    bg_cntrl_edited_image = edited_image * mask[None] + unedited_image * bg_mask[None] 
                # TODO
                # Promt testing
                # -> SDXLControlNetInpainPipeline
                # Dynamic Ref_image
                self.datamanager.train_data[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32) # [512 512 3]
                saving_image(self.datamanager.train_data[global_idx]["image"], global_idx)
        print("#############################")
        CONSOLE.print("Done Editing", style="bold yellow")
        print("#############################")
    
    
    @torch.no_grad()
    def image2latent(self, image):
        """Encode images to latents"""
        image = image * 2 - 1
        image = image.permute(2, 0, 1).unsqueeze(0) # torch.Size([1, 3, 512, 512]) -1~1
        latents = self.pipe.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    def depth2disparity(self, depth):
        """
        Args: depth numpy array [1 512 512]
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=0)
        return disparity_map[None]
    
    def depth2disparity_torch(self, depth):
        """
        Args: depth torch tensor
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / torch.max(disparity) # 0.00233~1
        disparity_map = torch.concatenate([disparity_map, disparity_map, disparity_map], dim=0)
        return disparity_map[None]

    def update_datasets(self, index, unedited_image, depth, latent, mask):
        """Save mid results"""
        self.datamanager.train_data[index]["unedited_image"] = unedited_image 
        self.datamanager.train_data[index]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy()
        self.datamanager.train_data[index]["z_0_image"] = latent.cpu().to(torch.float32).numpy()
        if mask is not None:
            self.datamanager.train_data[index]["mask_image"] = mask 

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step) # camera, data
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError