import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_path_or_dir, transform=None):
        self.transform = transform
        self.image_paths = []

        if os.path.isdir(image_path_or_dir):
            self.image_paths = [
                os.path.join(image_path_or_dir, f)
                for f in os.listdir(image_path_or_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
            ]
        elif os.path.isfile(image_path_or_dir) and image_path_or_dir.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            self.image_paths = [image_path_or_dir]
        else:
            raise ValueError(f"none: {image_path_or_dir}")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(img_path)


