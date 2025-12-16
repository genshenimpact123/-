#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字图像处理平台 - 主程序入口
修复色彩显示问题
Author: Student
Date: 2024
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image

class ColorFixManager:
    """色彩修复管理器"""
    
    @staticmethod
    def fix_cv2_functions():
        """
        修正OpenCV函数以正确处理色彩
        """
        print("初始化色彩修复管理器...")
        
        # 保存原始函数
        ColorFixManager._original_imread = cv2.imread
        ColorFixManager._original_imwrite = cv2.imwrite
        
        @staticmethod
        def fixed_imread(file_path, flags=cv2.IMREAD_COLOR):
            """
            修正的imread函数，自动转换为RGB
            """
            # 调用原始函数
            image = ColorFixManager._original_imread(file_path, flags)
            
            if image is not None:
                # 如果是彩色图像且不是灰度模式
                if flags != cv2.IMREAD_GRAYSCALE and len(image.shape) == 3:
                    # 自动转换为RGB
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
        
        @staticmethod
        def fixed_imwrite(file_path, image, params=None):
            """
            修正的imwrite函数，自动处理色彩空间
            """
            if image is None:
                return False
            
            save_image = image.copy()
            
            # 如果是彩色图像
            if len(save_image.shape) == 3:
                # 检查是否为RGB（通过简单的启发式判断）
                # 如果是RGB，转换为BGR
                if ColorFixManager._is_likely_rgb(save_image):
                    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            
            if params is None:
                return ColorFixManager._original_imwrite(file_path, save_image)
            else:
                return ColorFixManager._original_imwrite(file_path, save_image, params)
        
        # 替换函数
        cv2.imread = fixed_imread
        cv2.imwrite = fixed_imwrite
        
        print("色彩修复完成：所有图像将自动正确处理色彩空间")
    
    @staticmethod
    def _is_likely_rgb(image):
        """
        判断图像是否为RGB格式（而不是BGR）
        使用启发式方法
        """
        if len(image.shape) != 3:
            return False
        
        # 方法1：检查第一个像素
        b, g, r = image[0, 0]
        
        # 在自然图像中，通常红色比蓝色多一些
        # 如果蓝色比红色大很多，可能是BGR
        if b > r * 1.5:
            return False
        
        # 方法2：检查整个图像的平均值
        avg_b = np.mean(image[:, :, 0])
        avg_r = np.mean(image[:, :, 2])
        
        # 如果蓝色平均值明显大于红色，可能是BGR
        if avg_b > avg_r * 1.2:
            return False
        
        return True
    
    @staticmethod
    def test_color_fix():
        """测试色彩修复"""
        print("\n测试色彩修复功能...")
        
        # 创建测试图像
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # 纯红色 (RGB格式)
        
        # 保存测试图像
        test_path = "test_color_fix.png"
        cv2.imwrite(test_path, test_image)
        
        # 重新加载
        loaded = cv2.imread(test_path)
        
        # 检查颜色
        if loaded is not None:
            # 第一个像素应该是红色 (255, 0, 0) 在RGB中
            pixel = loaded[0, 0]
            print(f"测试像素值: {pixel}")
            
            if pixel[0] > 200 and pixel[1] < 50 and pixel[2] < 50:
                print("✓ 色彩修复测试通过")
            else:
                print("✗ 色彩修复可能有问题")
        
        # 清理
        if os.path.exists(test_path):
            os.remove(test_path)

class ImageProcessingPlatform:
    """图像处理平台主类"""
    
    def __init__(self):
        # 首先应用色彩修复
        ColorFixManager.fix_cv2_functions()
        
        self.root = tk.Tk()
        self.setup_application()
        
    def setup_application(self):
        """配置应用程序"""
        # 设置窗口属性
        self.root.title("数字图像处理平台")
        self.root.geometry("1400x800")
        
        # 设置图标
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # 设置样式
        self.setup_styles()
        
        # 确保必要的目录存在
        from utils.file_operations import ensure_directory
        ensure_directory("images")
        ensure_directory("output")
        
        # 创建主窗口
        from gui.main_window import MainWindow
        self.main_window = MainWindow(self.root)
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_styles(self):
        """配置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置标签样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Normal.TLabel', font=('Arial', 10))
        
        # 配置按钮样式
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'))
        style.configure('Success.TButton', font=('Arial', 10))
        style.configure('Warning.TButton', font=('Arial', 10))
        
        # 配置框架样式
        style.configure('Group.TLabelframe', font=('Arial', 11, 'bold'))
        
    def on_closing(self):
        """关闭应用程序时的处理"""
        if messagebox.askokcancel("退出", "确定要退出图像处理平台吗？"):
            self.root.destroy()
            
    def run(self):
        """运行应用程序"""
        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("错误", f"程序运行出错: {str(e)}")
            sys.exit(1)

def main():
    """主函数"""
    print("=" * 60)
    print("数字图像处理平台")
    print("版本: 1.0.0 (色彩修复版)")
    print("作者: Student")
    print("=" * 60)
    
    # 运行色彩修复测试
    ColorFixManager.test_color_fix()
    
    try:
        # 创建并运行应用程序
        app = ImageProcessingPlatform()
        app.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序启动失败: {e}")
        messagebox.showerror("启动错误", f"程序启动失败:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()