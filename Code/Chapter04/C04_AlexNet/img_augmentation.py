"""
文件名: Code/Chapter04/C04_AlexNet/img_augmentation.py
创建时间: 2023/3/24 19:23 下午

"""
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def img_aug(img_path=None):
    """
    图像增强
    """
    trans = [transforms.RandomRotation(degrees=90),
             transforms.RandomHorizontalFlip(p=0.9),
             transforms.RandomCrop(size=(928, 992), padding=(300, 200)),
             transforms.ColorJitter(brightness=1., contrast=0.2)]

    img = Image.open(img_path)
    img_array = np.array(img)
    plt.subplots(2, 3)

    plt.subplot(2, 3, 1)
    plt.imshow(img_array)  # 原始图片
    print(img_array.shape)
    img = transforms.PILToTensor()(img)
    plt.xlabel("Original", fontsize=10)
    plt.yticks([])
    plt.xticks([])

    plt.subplot(2, 3, 2)
    # 旋转给定的图像
    img_norm = trans[0](img)
    img_norm = transforms.ToPILImage()(img_norm)
    plt.imshow(img_norm)
    plt.xlabel("Rotation", fontsize=10)
    plt.yticks([])
    plt.xticks([])

    plt.subplot(2, 3, 3)
    # 以给定的概率随机水平翻转给定的图像
    img_norm = trans[1](img)
    img_norm = transforms.ToPILImage()(img_norm)
    plt.imshow(img_norm)
    plt.xlabel("HorizontalFlip", fontsize=10)
    plt.yticks([])
    plt.xticks([])

    plt.subplot(2, 3, 4)
    # 在随机位置裁剪给定的图像。
    img_norm = trans[2](img)
    print(img_norm.shape)
    img_norm = transforms.ToPILImage()(img_norm)
    plt.imshow(img_norm)
    plt.xlabel("Crop", fontsize=10)
    plt.yticks([])
    plt.xticks([])

    plt.subplot(2, 3, 5)
    img_norm = trans[3](img)
    print(img_norm.shape)
    img_norm = transforms.ToPILImage()(img_norm)
    plt.imshow(img_norm)
    plt.xlabel("ColorJitter", fontsize=10)
    plt.yticks([])
    plt.xticks([])

    plt.subplot(2, 3, 6)
    img_norm = transforms.Compose(trans)(img)
    img_norm = transforms.ToPILImage()(img_norm)
    plt.imshow(img_norm)
    plt.xlabel("Compose", fontsize=10)
    plt.yticks([])
    plt.xticks([])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    img_aug('../../data/dufu.png')
