import torch
import torch.nn as nn
import torchvision.models as models
import kornia.color as kcolor
from watson_vgg import WatsonDistanceVgg

def color_consistency_loss(original, reconstructed):
    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)

    if torch.isnan(original).any() or torch.isinf(original).any():
        raise ValueError("Original image contains NaN or Inf values")
    if torch.isnan(reconstructed).any() or torch.isinf(reconstructed).any():
        raise ValueError("Reconstructed image contains NaN or Inf values")

    original_lab = kcolor.rgb_to_lab(original)  # L: [0, 100], a/b: [-128, 127]
    reconstructed_lab = kcolor.rgb_to_lab(reconstructed)

    if torch.isnan(original_lab).any() or torch.isinf(original_lab).any():
        raise ValueError("LAB conversion of original image produced NaN or Inf")
    if torch.isnan(reconstructed_lab).any() or torch.isinf(reconstructed_lab).any():
        raise ValueError("LAB conversion of reconstructed image produced NaN or Inf")

    mse_loss = nn.MSELoss()
    loss = mse_loss(original_lab, reconstructed_lab)

    return loss


class LossPrep(nn.Module):
    def __init__(self, device):
        super(LossPrep, self).__init__()
        # add perceptive loss
        loss_percep = WatsonDistanceVgg(reduction='sum')
        loss_percep.load_state_dict(torch.load('./utils/rgb_watson_vgg_trial0.pth', map_location='cpu'))
        loss_percep = loss_percep.to(device)
        self.loss_per = lambda pred_img, gt_img: loss_percep((1 + pred_img) / 2.0, (1 + gt_img) / 2.0) / pred_img.shape[
            0]

    def __call__(self, pred_img_tensor, gt_img_tensor):
        lossP = self.loss_per(pred_img_tensor, gt_img_tensor)

        return lossP


def random_noise_loss(re_watermark_zero):
    mean = torch.mean(re_watermark_zero)
    var = torch.var(re_watermark_zero, unbiased=False)

    mean_loss = torch.mean(mean ** 2)
    var_loss = torch.mean((var - 1) ** 2)

    total_loss = mean_loss + var_loss
    return total_loss


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def adversarial_loss(discriminator, original, reconstructed):
    bce_loss = nn.BCELoss()
    real_labels = torch.ones(original.size(0), 1, 1, 1, device=original.device)
    fake_labels = torch.zeros(reconstructed.size(0), 1, 1, 1, device=reconstructed.device)

    real_output = discriminator(original)
    real_loss = bce_loss(real_output, real_labels)

    fake_output = discriminator(reconstructed)
    fake_loss = bce_loss(fake_output, fake_labels)

    return real_loss + fake_loss
