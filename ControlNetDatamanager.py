import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union
import torch
from tqdm import tqdm
from copy import deepcopy
from rich.progress import Console

CONSOLE = Console(width=120)

@dataclass
class DataManagerConfig:
    """Configuration for the custom DataManager."""

    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""
    subset_num: int = 4
    """Subset sample split number."""
    sampled_views_every_subset: int = 10
    """Number of images sampled in each subset split."""
    load_all: bool = False
    """Set to True to use all images."""


class DataManager:
    """Custom data manager for handling image arrays."""

    def __init__(self,
                 config: DataManagerConfig,
                 train_images: List[torch.Tensor],
                 eval_images: List[torch.Tensor],
                 device: Union[torch.device, str] = "cpu",
                 **kwargs):
        self.config = config
        self.device = device
        self.train_images = train_images
        self.eval_images = eval_images

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
                # Assuming data is a tensor representing an image
                self.train_data.append({'image': data, 'image_idx': i})
            self.train_unseen_cameras = list(range(self.config.subset_num * self.config.sampled_views_every_subset))

    def cache_images(self):
        """Caches the train and eval images in memory (useful if large images)."""
        cached_train = []
        cached_eval = []

        CONSOLE.log("Caching train images")
        for i in tqdm(range(len(self.train_data)), leave=False):
            data = deepcopy(self.train_data[i])
            data["image"] = data["image"].to(self.device)
            cached_train.append(data)

        CONSOLE.log("Caching eval images")
        for i in tqdm(range(len(self.eval_images)), leave=False):
            data = deepcopy(self.eval_images[i])
            cached_eval.append(data)

        return cached_train, cached_eval

    def next_train(self) -> Dict:
        """Returns the next training batch."""
        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))

        # Re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_data))]

        data = deepcopy(self.train_data[image_idx])
        data["image"] = data["image"].to(self.device)

        return data
