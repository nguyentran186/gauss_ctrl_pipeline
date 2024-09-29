import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode
from einops import rearrange
import glob
from diffusers.utils import USE_PEFT_BACKEND
import cv2
from transformers import pipeline

def read_depth2disparity(depth_dir):
    depth_paths = sorted(glob.glob(depth_dir + '/*.npy'))
    disparity_list = []
    for depth_path in depth_paths:
        depth = np.load(depth_path) # [512,512,1] 
        
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        # disparity_map = disparity_map.astype(np.uint8)[:,:,0]
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=2)
        disparity_list.append(disparity_map[None]) 

    detected_maps = np.concatenate(disparity_list, axis=0)
    
    control = torch.from_numpy(detected_maps.copy()).float()
    return rearrange(control, 'f h w c -> f c h w')

def compute_attn(attn, query, key, value, video_length, ref_frame_index, attention_mask):
    key_ref_cross = rearrange(key, "(b f) d c -> b f d c", f=video_length)
    key_ref_cross = key_ref_cross[:, ref_frame_index]
    key_ref_cross = rearrange(key_ref_cross, "b f d c -> (b f) d c")
    value_ref_cross = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    value_ref_cross = value_ref_cross[:, ref_frame_index]
    value_ref_cross = rearrange(value_ref_cross, "b f d c -> (b f) d c")

    key_ref_cross = attn.head_to_batch_dim(key_ref_cross)
    value_ref_cross = attn.head_to_batch_dim(value_ref_cross)
    attention_probs = attn.get_attention_scores(query, key_ref_cross, attention_mask)
    hidden_states_ref_cross = torch.bmm(attention_probs, value_ref_cross) 
    return hidden_states_ref_cross

class CrossViewAttnProcessor:
    def __init__(self, self_attn_coeff, unet_chunk_size=2, num_ref=1):
        self.unet_chunk_size = unet_chunk_size
        self.self_attn_coeff = self_attn_coeff
        self.num_ref = num_ref

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):

        residual = hidden_states
        
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        query = attn.head_to_batch_dim(query)
        # Sparse Attention
        if not is_cross_attention:
            ################## Perform self attention
            key_self = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
            hidden_states_self = torch.bmm(attention_probs, value_self)
            #######################################

            video_length = key.size()[0] // self.unet_chunk_size

            list_ref_frame_index = []
            hidden_state_ref = []
            for i in range(self.num_ref):
                list_ref_frame_index.append([i] * video_length)
                if i == (self.num_ref-1):
                    continue
                hidden_state_ref.append(
                    compute_attn(attn, query, key, value, video_length, list_ref_frame_index[i], attention_mask)
                )

            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, list_ref_frame_index[-1]]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, list_ref_frame_index[-1]]
            value = rearrange(value, "b f d c -> (b f) d c")

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states_ref_last = torch.bmm(attention_probs, value)
        
        hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff) * torch.mean(torch.stack(hidden_state_ref+[hidden_states_ref_last]), dim=0) if not is_cross_attention else hidden_states_ref_last
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def saving_image(image, name):
    image = image.cpu().numpy()
    image = (image * 255).clip(0, 255).astype('uint8')
    image = cv2.resize(image, (1008, 756))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(name) + "_edited.png", image)
    
