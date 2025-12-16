# test_color.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def test_image_color():
    """测试图像色彩问题"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    file_path = filedialog.askopenfilename(
        title="选择测试图像",
        filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        return
    
    print("=" * 60)
    print("图像色彩测试")
    print("=" * 60)
    
    # 方法1：OpenCV直接加载
    img_cv = cv2.imread(file_path)
    if img_cv is not None:
        print(f"\n1. OpenCV直接加载:")
        print(f"   形状: {img_cv.shape}")
        print(f"   第一个像素 (BGR): {img_cv[0, 0]}")
        print(f"   像素范围: [{img_cv.min()}, {img_cv.max()}]")
    
    # 方法2：OpenCV加载后转RGB
    if img_cv is not None and len(img_cv.shape) == 3:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        print(f"\n2. OpenCV加载后转RGB:")
        print(f"   第一个像素 (RGB): {img_rgb[0, 0]}")
    
    # 方法3：PIL加载
    from PIL import Image
    pil_img = Image.open(file_path)
    print(f"\n3. PIL加载:")
    print(f"   模式: {pil_img.mode}")
    print(f"   尺寸: {pil_img.size}")
    
    if pil_img.mode == 'RGB':
        pil_array = np.array(pil_img)
        print(f"   第一个像素 (RGB): {pil_array[0, 0]}")
    
    # 显示图像对比
    if img_cv is not None:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # OpenCV直接显示（BGR）
        axes[0].imshow(img_cv)
        axes[0].set_title("OpenCV直接显示 (BGR)")
        axes[0].axis('off')
        
        # 转换为RGB后显示
        if len(img_cv.shape) == 3:
            axes[1].imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            axes[1].set_title("转换为RGB后显示")
            axes[1].axis('off')
        
        # PIL显示
        axes[2].imshow(pil_img)
        axes[2].set_title("PIL显示")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("\n" + "=" * 60)
    print("结论:")
    print("- 如果图1颜色不对，图2和图3颜色正确：OpenCV BGR问题")
    print("- 如果三张图颜色都不对：图像文件本身有问题")
    print("- 如果三张图颜色都正确：加载代码有问题")
    print("=" * 60)

if __name__ == "__main__":
    test_image_color()