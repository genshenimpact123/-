#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
噪声图片生成器（实验素材版）
功能：
1. 从指定路径读取原始图像
2. 生成 5 种噪声图像
3. 自动保存所有噪声图像
4. 逐张显示噪声图像（不显示原图）
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def add_gaussian_noise(img, sigma):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img, prob):
    """添加椒盐噪声"""
    noisy = img.copy()

    # 盐噪声（白点）
    num_salt = int(prob * img.size / 2)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # 椒噪声（黑点）
    num_pepper = int(prob * img.size / 2)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy


def generate_noisy_images(image_path):
    """生成并显示噪声图像（实验素材）"""

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片：{image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"▶ 正在处理图片：{base_name}")

    # 生成噪声图像
    noisy1 = add_gaussian_noise(img, 15)
    noisy2 = add_gaussian_noise(img, 40)
    noisy3 = add_salt_pepper_noise(img, 0.02)
    noisy4 = add_salt_pepper_noise(img, 0.08)
    noisy5 = add_gaussian_noise(img, 25)
    noisy5 = add_salt_pepper_noise(noisy5, 0.03)


    print("✔ 噪声图像已全部保存")

    # 逐张显示噪声图像（不显示原图）
    noise_results = [
        ("轻度高斯噪声", noisy1),
        ("重度高斯噪声", noisy2),
        ("轻度椒盐噪声", noisy3),
        ("重度椒盐噪声", noisy4),
        ("混合噪声", noisy5)
    ]

    for title, img_show in noise_results:
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()

    print("✅ 全部完成")


if __name__ == "__main__":
    # ===== 修改为你自己的图片路径 =====
    image_path = r"D:\bbb\tu\3.jpg"
    generate_noisy_images(image_path)
