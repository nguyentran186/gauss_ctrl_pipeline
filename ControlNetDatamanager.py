import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union
import torch
from tqdm import tqdm
from copy import deepcopy
import cv2  # OpenCV for image loading
from rich.progress import Console

CONSOLE = Console(width=120)

@dataclass
class DataManagerConfig:
    """Configuration for the custom DataManager."""
    patch_size: int = 32
    subset_num: int = 4
    sampled_views_every_subset: int = 10
    load_all: bool = False

class DataManager:
    """Custom data manager for handling image file paths and loading them as tensors."""

    def __init__(self,
                 config: DataManagerConfig,
                 train_image_paths: List[str],
                 eval_image_paths: List[str],
                 image_size: Tuple[int, int] = (512, 512),  # Image size to resize
                 device: Union[torch.device, str] = "cpu",
                 **kwargs):
        self.config = config
        self.device = device
        self.image_size = image_size
        self.train_image_paths = train_image_paths
        self.eval_image_paths = eval_image_paths

        self.sample_idx = []
        self.step_every = 1
        self.edited_image_dict = {}

        # Sample data
        if len(self.train_image_paths) <= self.config.subset_num * self.config.sampled_views_every_subset or self.config.load_all:
            self.train_data = self._load_images(self.train_image_paths)
            self.train_unseen_cameras = list(range(len(self.train_data)))
        else:
            view_num = len(self.train_image_paths)
            anchors = list(range(0, view_num, view_num // self.config.subset_num))[:4] + [view_num]
            sampled_indices = []
            for idx in range(len(anchors[:-1])):
                cur_anchor = anchors[idx]
                next_anchor = anchors[idx+1]
                selected = sorted(random.sample(list(range(cur_anchor, next_anchor)), self.config.sampled_views_every_subset))
                sampled_indices += selected

            self.train_data_temp = self._load_images([self.train_image_paths[i] for i in sampled_indices])
            self.train_data = []
            for i, data in enumerate(self.train_data_temp):
                self.train_data.append({'image': data, 'image_idx': i})
            self.train_unseen_cameras = list(range(self.config.subset_num * self.config.sampled_views_every_subset))

    def _load_images(self, image_paths: List[str]) -> List[torch.Tensor]:
        """Loads images from paths using OpenCV and converts them to PyTorch tensors."""
        images = []
        for path in image_paths:
            img = cv2.imread(path)  # Load image with OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, self.image_size)  # Resize image
            img = img.astype('float32') / 255.0  # Normalize image
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # Convert to PyTorch tensor (CHW)
            images.append(img_tensor)
        return images

    def cache_images(self):
        """Caches the train and eval images in memory."""
        cached_train = []
        cached_eval = []

        CONSOLE.log("Caching train images")
        for i in tqdm(range(len(self.train_data)), leave=False):
            data = deepcopy(self.train_data[i])
            data["image"] = data["image"].to(self.device)
            cached_train.append(data)

        CONSOLE.log("Caching eval images")
        for i in tqdm(range(len(self.eval_image_paths)), leave=False):
            img = cv2.imread(self.eval_image_paths[i])  # Load eval image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            img = img.astype('float32') / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).to(self.device)
            cached_eval.append(img_tensor)

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
