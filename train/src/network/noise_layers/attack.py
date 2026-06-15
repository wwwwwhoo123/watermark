import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms


# 亮度攻击类
class BrightnessAttacker(nn.Module):
    def __init__(self, brightness=0.2):
        """
        初始化亮度攻击
        参数:
            brightness (float): 亮度调整因子，默认0.2（对应PIL的enhance因子）
        """
        super(BrightnessAttacker, self).__init__()
        self.brightness = brightness

    def adjust_brightness(self, image):
        """
        调整图像亮度
        参数:
            image (torch.Tensor): 输入图像，范围应为 [0, 1]
        返回:
            torch.Tensor: 调整亮度后的图像
        """
        # PIL enhance 的因子 > 1 增亮，< 1 减暗，这里转换为加法操作
        brightness_factor = self.brightness - 1.0
        out = image + brightness_factor
        return torch.clamp(out, 0, 1)

    def attack(self, image_paths, out_paths, multi=False):
        """
        执行亮度攻击并保存结果
        参数:
            image_paths (list): 输入图像路径列表
            out_paths (list): 输出图像路径列表
            multi (bool): 是否允许多次处理同一输出路径
        """
        for img_path, out_path in zip(image_paths, out_paths):
            if os.path.exists(out_path) and not multi:
                continue

            # 加载图像并转换为 Tensor
            img = Image.open(img_path).convert('RGB')
            transform = transforms.ToTensor()  # 转换为 [0, 1] 范围
            img_tensor = transform(img).unsqueeze(0)  # 添加 batch 维度

            # 调整亮度
            attacked_tensor = self.adjust_brightness(img_tensor)

            # 转换为 PIL 图像并保存
            attacked_img = transforms.ToPILImage()(attacked_tensor.squeeze(0))
            attacked_img.save(out_path)

    def forward(self, image):
        return self.adjust_brightness(image)


# 对比度攻击类
class ContrastAttacker(nn.Module):
    def __init__(self, contrast=0.2):
        """
        初始化对比度攻击
        参数:
            contrast (float): 对比度调整因子，默认0.2（对应PIL的enhance因子）
        """
        super(ContrastAttacker, self).__init__()
        self.contrast = contrast

    def adjust_contrast(self, image):
        """
        调整图像对比度
        参数:
            image (torch.Tensor): 输入图像，范围应为 [0, 1]
        返回:
            torch.Tensor: 调整对比度后的图像
        """
        mean = torch.mean(image, dim=(1, 2, 3), keepdim=True)
        out = (image - mean) * self.contrast + mean
        return torch.clamp(out, 0, 1)

    def attack(self, image_paths, out_paths, multi=False):
        """
        执行对比度攻击并保存结果
        参数:
            image_paths (list): 输入图像路径列表
            out_paths (list): 输出图像路径列表
            multi (bool): 是否允许多次处理同一输出路径
        """
        for img_path, out_path in zip(image_paths, out_paths):
            if os.path.exists(out_path) and not multi:
                continue

            # 加载图像并转换为 Tensor
            img = Image.open(img_path).convert('RGB')
            transform = transforms.ToTensor()  # 转换为 [0, 1] 范围
            img_tensor = transform(img).unsqueeze(0)  # 添加 batch 维度

            # 调整对比度
            attacked_tensor = self.adjust_contrast(img_tensor)

            # 转换为 PIL 图像并保存
            attacked_img = transforms.ToPILImage()(attacked_tensor.squeeze(0))
            attacked_img.save(out_path)

    def forward(self, image):
        return self.adjust_contrast(image)


# 旋转攻击类
class RotateAttacker(nn.Module):
    def __init__(self, degree=30, expand=1):
        """
        初始化旋转攻击
        参数:
            degree (float): 旋转角度（度），默认30
            expand (int): 是否扩展图像以适应旋转，默认1（True）
        """
        super(RotateAttacker, self).__init__()
        self.degree = degree
        self.expand = bool(expand)

    def rotate(self, image):
        """
        旋转图像
        参数:
            image (torch.Tensor): 输入图像，范围应为 [0, 1]
        返回:
            torch.Tensor: 旋转并调整大小后的图像
        """
        # 转换为 PIL 图像以使用 rotate
        pil_img = transforms.ToPILImage()(image.squeeze(0))
        rotated_img = pil_img.rotate(self.degree, expand=self.expand)

        # 调整大小到 512x512
        resized_img = rotated_img.resize((512, 512))

        # 转换回 Tensor
        return transforms.ToTensor()(resized_img).unsqueeze(0)

    def attack(self, image_paths, out_paths, multi=False):
        """
        执行旋转攻击并保存结果
        参数:
            image_paths (list): 输入图像路径列表
            out_paths (list): 输出图像路径列表
            multi (bool): 是否允许多次处理同一输出路径
        """
        for img_path, out_path in zip(image_paths, out_paths):
            if os.path.exists(out_path) and not multi:
                continue

            # 加载图像并转换为 Tensor
            img = Image.open(img_path).convert('RGB')
            transform = transforms.ToTensor()  # 转换为 [0, 1] 范围
            img_tensor = transform(img).unsqueeze(0)  # 添加 batch 维度

            # 旋转并调整大小
            attacked_tensor = self.rotate(img_tensor)

            # 转换为 PIL 图像并保存
            attacked_img = transforms.ToPILImage()(attacked_tensor.squeeze(0))
            attacked_img.save(out_path)

    def forward(self, image):
        return self.rotate(image)


# 示例用法
if __name__ == "__main__":
    import os

    image_paths = [r"E:\AProject\awmencoder_irf\dataset\coco_test\00000.png"]
    out_paths = ["output_brightness.jpg", "output_contrast.jpg", "output_rotate.jpg"]

    # 亮度攻击
    brightness_attacker = BrightnessAttacker(brightness=0.5)  # 增亮
    brightness_attacker.attack(image_paths, [out_paths[0]])

    # 对比度攻击
    contrast_attacker = ContrastAttacker(contrast=0.5)  # 增加对比度
    contrast_attacker.attack(image_paths, [out_paths[1]])

    # 旋转攻击
    rotate_attacker = RotateAttacker(degree=45, expand=1)
    rotate_attacker.attack(image_paths, [out_paths[2]])
