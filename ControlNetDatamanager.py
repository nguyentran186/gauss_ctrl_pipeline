import random
from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from rich.progress import Console
import os
from PIL import Image

CONSOLE = Console(width=120)

@dataclass
class DataManagerConfig:
    """Configuration for the custom DataManager."""
    patch_size: int = 32
    subset_num: int = 4
    sampled_views_every_subset: int = 10
    load_all: bool = False

class DataManager:
    """Custom data manager for handling image arrays loaded from folder paths."""

    def __init__(self,
                 config: DataManagerConfig,
                 data_folder: str,
                 device: Union[str] = "cpu",
                 **kwargs):
        self.config = config
        self.device = device
        self.train_images = self.load_images_from_folder(data_folder)

        self.sample_idx = []
        self.step_every = 1
        self.edited_image_dict = {}

        # Sample data
        if len(self.train_images) <= self.config.subset_num * self.config.sampled_views_every_subset or self.config.load_all:
            self.train_data = self.train_images
            self.train_unseen_cameras = list(range(len(self.train_data)))
        else:
            view_num = len(self.train_images)
            anchors = list(range(0, view_num, view_num // self.config.subset_num))[:4] + [view_num]
            sampled_indices = []
            for idx in range(len(anchors[:-1])):
                cur_anchor = anchors[idx]
                next_anchor = anchors[idx+1]
                selected = sorted(random.sample(list(range(cur_anchor, next_anchor)), self.config.sampled_views_every_subset))
                sampled_indices += selected

            self.train_data_temp = [self.train_images[i] for i in sampled_indices]
            self.train_data = []
            for i, data in enumerate(self.train_data_temp):
                # Assuming data contains 'rgb' and 'depth' fields
                self.train_data.append({
                    'image': data['image'],
                    'z_0_image': data['z_0_image'], 
                    'depth_image': data['depth_image'], 
                    'unedited_image': data['unedited_image'], 
                    'image_idx': i})
            self.train_unseen_cameras = list(range(self.config.subset_num * self.config.sampled_views_every_subset))

    def load_images_from_folder(self, folder_path: str) -> List[Dict[str, np.ndarray]]:
        """Loads and matches RGB and depth images from the specified folder."""
        image_folder = os.path.join(folder_path, 'images')
        # depth_folder = os.path.join(folder_path, 'depth')
        depth_folder = '/home/ubuntu/workspace/bhrc/nam/gauss_ctrl_pipeline/data/statue_inpainted/depth'
        mask_folder = os.path.join(folder_path, 'mask')

        # Get the list of image and depth files
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])
        depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith(('.png', '.jpg'))])
        mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg'))])

        assert len(image_files) == len(depth_files), "Mismatch between image and depth file counts."

        image_depth_pairs = []
        
        for image_idx, (img_file, depth_file) in enumerate(zip(image_files, depth_files)):
            img_name, depth_name = img_file.split('.')[0], depth_file.split('.')[0]
            assert img_name == depth_name, f"Image and depth file names do not match: {img_name} vs {depth_name}"

            # Load images and convert to NumPy arrays
            rgb_image = np.array(Image.open(os.path.join(image_folder, img_file)).convert('RGB'))
            depth_image = np.array(Image.open(os.path.join(depth_folder, depth_file)).convert('L'))
            mask_image = np.array(Image.open(os.path.join(mask_folder, mask_files[image_idx])).convert('L'))

            # Add depth dimension explicitly
            depth_image = np.expand_dims(depth_image, axis=-1)
            image_depth_pairs.append({'z_0_image': rgb_image, 
                                      'image': rgb_image,
                                      'depth_image': depth_image, 
                                      'unedited_image': rgb_image, 
                                      'mask_image': mask_image,
                                      'image_idx': image_idx,
                                      'image_name': img_name,
                                      })

        return image_depth_pairs


    def cache_images(self):
        """Caches the train and eval images in memory."""
        cached_train = []
        cached_eval = []

        CONSOLE.log("Caching train images")
        for i in tqdm(range(len(self.train_data)), leave=False):
            data = deepcopy(self.train_data[i])
            cached_train.append(data)

        return cached_train

    def next_train(self) -> Dict:
        """Returns the next training batch."""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))

        # Re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_data))]

        data = deepcopy(self.train_data[image_idx])

        return data
