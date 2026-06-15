import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from torchvision import transforms


class Rotate(nn.Module):
    def __init__(self, degree=30, expand=False):
        super(Rotate, self).__init__()
        self.degree = degree
        self.expand = expand

    def forward(self, image_and_cover):
        img, cover_image = image_and_cover

        img = TF.rotate(
            img,
            angle=self.degree,
            interpolation=TF.InterpolationMode.BILINEAR,
            expand=self.expand
        )

        img = TF.resize(
            img,
            size=[512, 512],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True
        )

        return img


if __name__ == "__main__":
    # 读取图像并转换为tensor
    image_path = r"E:\AProject\awmencoder_irf\dataset\coco_test\00000.png"  # 替换成你的图像路径
    cover_image_path = r"E:\AProject\awmencoder_irf\dataset\coco_test\00000.png"  # 替换成你的封面图像路径

    # 使用PIL加载图像
    image_pil = Image.open(image_path)
    cover_image_pil = Image.open(cover_image_path)

    # 将PIL图像转换为Tensor
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image_pil)
    cover_image_tensor = to_tensor(cover_image_pil)

    # 创建旋转攻击实例
    rotate_attack = Rotate(degree=90)  # 旋转45度并扩展图像

    # 执行旋转攻击
    attacked_image_tensor = rotate_attack((image_tensor, cover_image_tensor))

    # 将tensor转换回PIL图像以显示
    original_image_pil = transforms.ToPILImage()(image_tensor)
    attacked_image_pil = transforms.ToPILImage()(attacked_image_tensor)

    # 显示原图和攻击后图像
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 显示原图
    axes[0].imshow(original_image_pil)
    axes[0].set_title("原始图像")
    axes[0].axis("off")

    # 显示攻击后的图像
    axes[1].imshow(attacked_image_pil)
    axes[1].set_title("攻击后图像 (旋转)")
    axes[1].axis("off")

    plt.show()
