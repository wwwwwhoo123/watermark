import random
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np


class WatermarkAugmenter:
    def __init__(self, watermark_path, batch_size, device=None):
        self.base_watermark = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])(
            Image.open(watermark_path)
        ).unsqueeze(0).to(device)

        self.batch_size = batch_size
        self.transform_idx = 0
        self.device = device
        self.current_epoch = 0
        self.epoch_threshold = 40

        self.transforms = [
            self.identity,
            self.rotation120,
            self.rotation240,
        ]

    def update_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch < self.epoch_threshold:
            if self.invert_colors not in self.transforms:
                self.transforms.append(self.invert_colors)
        else:
            if self.invert_colors in self.transforms:
                self.transforms.remove(self.invert_colors)

    def identity(self, watermark):
        return watermark

    def invert_colors(self, watermark):
        return -0.5 * watermark

    def slight_rotation(self, watermark):
        angle = random.uniform(0, 30)
        return transforms.functional.rotate(watermark, angle)

    def rotation60(self, watermark):
        return transforms.functional.rotate(watermark, 60)

    def rotation90(self, watermark):
        return transforms.functional.rotate(watermark, 90)

    def rotation120(self, watermark):
        return transforms.functional.rotate(watermark, 120)

    def rotation150(self, watermark):
        return transforms.functional.rotate(watermark, 150)

    def rotation180(self, watermark):
        return transforms.functional.rotate(watermark, 180)

    def rotation210(self, watermark):
        return transforms.functional.rotate(watermark, 210)

    def rotation240(self, watermark):
        return transforms.functional.rotate(watermark, 240)

    def rotation300(self, watermark):
        return transforms.functional.rotate(watermark, 300)

    def invert_slight_rotation(self, watermark):
        watermark = -0.5 * watermark
        # angle = random.uniform(0, 30)
        return transforms.functional.rotate(watermark, 30)

    def contrast_jitter(self, watermark):
        factor = random.uniform(0.5, 2.5)
        return transforms.functional.adjust_contrast(watermark, factor)

    def brightness_jitter(self, watermark):
        factor = random.uniform(0.7, 2.3)
        return transforms.functional.adjust_brightness(watermark, factor)

    # def center_crop_pad(self, watermark):
    #     c, h, w = watermark.shape[1:]
    #     crop_size = int(min(h, w) * 0.8)
    #     cropped = transforms.functional.center_crop(watermark, [crop_size, crop_size])
    #     padded = transforms.functional.pad(cropped, padding=((w - crop_size) // 2, (h - crop_size) // 2))
    #     return padded

    def random_erasing(self, watermark):
        erased = watermark.clone()
        c, h, w = erased.shape[1:]
        erase_h = random.randint(h // 8, h // 4)
        erase_w = random.randint(w // 8, w // 4)
        x = random.randint(0, h - erase_h)
        y = random.randint(0, w - erase_w)
        erased[:, :, x:x + erase_h, y:y + erase_w] = 0
        return erased

    def slight_noise(self, watermark):
        noise = torch.randn_like(watermark) * 0.05
        return torch.clamp(watermark + noise, -1, 1)

    def horizontal_flip(self, watermark):
        return transforms.functional.hflip(watermark)

    def invert_horizontal_flip(self, watermark):
        watermark = -0.5 * watermark
        return transforms.functional.hflip(watermark)

    def vertical_flip(self, watermark):
        return transforms.functional.vflip(watermark)

    def invert_vertical_flip(self, watermark):
        watermark = -0.5 * watermark
        return transforms.functional.vflip(watermark)

    # def get_augmented_watermark(self):
    #     transform = random.choice(self.transforms)
    #     transformed_watermark = transform(self.base_watermark)
    #     return transformed_watermark.repeat(self.batch_size, 1, 1, 1)

    def get_augmented_watermark(self):
        transform = self.transforms[self.transform_idx % len(self.transforms)]
        # print(transform)
        self.transform_idx = (self.transform_idx + 1) % len(self.transforms)
        transformed_watermark = transform(self.base_watermark)
        return transformed_watermark.repeat(self.batch_size, 1, 1, 1)





