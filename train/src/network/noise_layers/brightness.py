import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as F


class Bright(nn.Module):
    def __init__(self, brightness=0.2):
        super(Bright, self).__init__()
        self.brightness = brightness

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        image = F.adjust_brightness(image, self.brightness)
        return image


if __name__ == "__main__":
    # 读取图像并转换为tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = r"E:\AProject\awmencoder_irf\dataset\coco_test\00000.png"  # 替换成你的图像路径
    cover_image_path = r"E:\AProject\awmencoder_irf\dataset\coco_test\00000.png"  # 替换成你的封面图像路径

    # 使用PIL加载图像
    image_pil = Image.open(image_path)
    cover_image_pil = Image.open(cover_image_path)

    # 将PIL图像转换为Tensor
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image_pil).unsqueeze(0).to(device)
    print(image_tensor.shape)
    cover_image_tensor = to_tensor(cover_image_pil).to(device)

    # 创建亮度增强攻击实例
    brightness_attack = Bright(brightness=0.5).to(device)

    # 执行亮度增强攻击
    attacked_image_tensor = brightness_attack([image_tensor, cover_image_tensor])
    print(attacked_image_tensor.is_cuda)

    # 将tensor转换回PIL图像以显示
    original_image_pil = transforms.ToPILImage()(image_tensor.squeeze(0))
    attacked_image_pil = transforms.ToPILImage()(attacked_image_tensor.squeeze(0))

    # 显示原图和攻击后图像
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 显示原图
    axes[0].imshow(original_image_pil)
    axes[0].set_title("原始图像")
    axes[0].axis("off")

    # 显示攻击后的图像
    axes[1].imshow(attacked_image_pil)
    axes[1].set_title("攻击后图像 (亮度增强)")
    axes[1].axis("off")

    plt.show()
