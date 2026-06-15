import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def save_image_comparison(original_images, reconstructed_images, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    original_images = (original_images + 1) / 2
    reconstructed_images = (reconstructed_images + 1) / 2
    comparison = torch.cat([original_images, reconstructed_images], dim=0)
    vutils.save_image(comparison, save_path, nrow=len(original_images), padding=2, normalize=False)