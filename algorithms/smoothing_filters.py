#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平滑滤波模块
实现均值滤波、中值滤波、高斯滤波等平滑滤波器
"""

import cv2
import numpy as np
from scipy import ndimage, signal

class SmoothingFilters:
    """平滑滤波类"""
    
    @staticmethod
    def mean_filter(image, kernel_size=3):
        """
        均值滤波
        
        参数:
            image: 输入图像
            kernel_size: 滤波器大小（必须是奇数）
            
        返回:
            滤波后的图像
        """
        if image is None:
            return None
            
        # 确保核大小为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # 应用均值滤波
        result = cv2.blur(image, (kernel_size, kernel_size))
        
        return result
    
    @staticmethod
    def median_filter(image, kernel_size=3):
        """
        中值滤波
        
        参数:
            image: 输入图像
            kernel_size: 滤波器大小（必须是奇数）
            
        返回:
            滤波后的图像
        """
        if image is None:
            return None
            
        # 确保核大小为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # 应用中值滤波
        result = cv2.medianBlur(image, kernel_size)
        
        return result
    
    @staticmethod
    def gaussian_filter(image, kernel_size=3, sigma=0):
        """
        高斯滤波
        
        参数:
            image: 输入图像
            kernel_size: 滤波器大小（必须是奇数）
            sigma: 高斯核的标准差，如果为0则自动计算
            
        返回:
            滤波后的图像
        """
        if image is None:
            return None
            
        # 确保核大小为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # 应用高斯滤波
        result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        return result
    
    @staticmethod
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        """
        双边滤波
        
        参数:
            image: 输入图像
            d: 每个像素邻域的直径
            sigma_color: 颜色空间的标准差
            sigma_space: 坐标空间的标准差
            
        返回:
            滤波后的图像
        """
        if image is None:
            return None
            
        # 应用双边滤波
        result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        return result
    
    @staticmethod
    def frequency_lowpass_filter(image, cutoff_ratio=0.5, filter_type='ideal'):
        """
        频率域低通滤波器
        
        参数:
            image: 输入图像
            cutoff_ratio: 截止频率比例（0-1）
            filter_type: 滤波器类型
                'ideal': 理想低通滤波器
                'gaussian': 高斯低通滤波器
                'butterworth': 巴特沃斯低通滤波器
                
        返回:
            滤波后的图像
        """
        if image is None:
            return None
            
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 傅里叶变换
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # 创建滤波器
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        if filter_type == 'ideal':
            # 理想低通滤波器
            mask = np.zeros((rows, cols, 2), np.float32)
            radius = int(min(rows, cols) * cutoff_ratio / 2)
            mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1
            
        elif filter_type == 'gaussian':
            # 高斯低通滤波器
            x = np.linspace(-ccol, ccol, cols)
            y = np.linspace(-crow, crow, rows)
            X, Y = np.meshgrid(x, y)
            D = np.sqrt(X**2 + Y**2)
            D0 = min(rows, cols) * cutoff_ratio / 2
            H = np.exp(-(D**2) / (2 * D0**2))
            mask = np.stack([H, H], axis=2)
            
        elif filter_type == 'butterworth':
            # 巴特沃斯低通滤波器
            x = np.linspace(-ccol, ccol, cols)
            y = np.linspace(-crow, crow, rows)
            X, Y = np.meshgrid(x, y)
            D = np.sqrt(X**2 + Y**2)
            D0 = min(rows, cols) * cutoff_ratio / 2
            n = 2  # 滤波器阶数
            H = 1 / (1 + (D / D0)**(2 * n))
            mask = np.stack([H, H], axis=2)
            
        else:
            raise ValueError(f"不支持的滤波器类型: {filter_type}")
        
        # 应用滤波器
        fshift = dft_shift * mask
        
        # 逆傅里叶变换
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        # 归一化
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        result = np.uint8(img_back)
        
        return result
    
    @staticmethod
    def wiener_filter(image, kernel_size=3, noise_variance=0.01):
        """
        维纳滤波
        
        参数:
            image: 输入图像
            kernel_size: 滤波器大小
            noise_variance: 噪声方差估计
            
        返回:
            滤波后的图像
        """
        if image is None:
            return None
            
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 将图像转换为浮点型
        gray_float = gray.astype(np.float32) / 255.0
        
        # 创建高斯模糊核
        kernel = cv2.getGaussianKernel(kernel_size, -1)
        kernel = kernel * kernel.T
        
        # 傅里叶变换
        img_fft = np.fft.fft2(gray_float)
        kernel_fft = np.fft.fft2(kernel, s=gray_float.shape)
        
        # 计算维纳滤波器
        kernel_abs_sq = np.abs(kernel_fft) ** 2
        wiener_filter = np.conj(kernel_fft) / (kernel_abs_sq + noise_variance / np.abs(img_fft + 1e-10))
        
        # 应用滤波器
        result_fft = img_fft * wiener_filter
        result = np.fft.ifft2(result_fft)
        result = np.abs(result)
        
        # 归一化
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def anisotropic_diffusion(image, iterations=10, kappa=50, gamma=0.1, option=1):
        """
        各向异性扩散滤波（Perona-Malik滤波）
        
        参数:
            image: 输入图像
            iterations: 迭代次数
            kappa: 传导系数
            gamma: 积分常数（0-1）
            option: 传导函数选项（1或2）
            
        返回:
            滤波后的图像
        """
        if image is None:
            return None
            
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 将图像转换为浮点型
        img = gray.astype(np.float32)
        
        # 初始化输出图像
        output = img.copy()
        
        # 定义传导函数
        if option == 1:
            def conduction_function(grad):
                return np.exp(-(grad / kappa) ** 2)
        elif option == 2:
            def conduction_function(grad):
                return 1 / (1 + (grad / kappa) ** 2)
        else:
            raise ValueError("option参数必须是1或2")
        
        # 迭代应用各向异性扩散
        for _ in range(iterations):
            # 计算梯度
            delta_n = np.roll(output, -1, axis=0) - output
            delta_s = np.roll(output, 1, axis=0) - output
            delta_e = np.roll(output, -1, axis=1) - output
            delta_w = np.roll(output, 1, axis=1) - output
            
            # 应用传导函数
            c_n = conduction_function(np.abs(delta_n))
            c_s = conduction_function(np.abs(delta_s))
            c_e = conduction_function(np.abs(delta_e))
            c_w = conduction_function(np.abs(delta_w))
            
            # 更新图像
            output = output + gamma * (
                c_n * delta_n + 
                c_s * delta_s + 
                c_e * delta_e + 
                c_w * delta_w
            )
        
        # 归一化
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    @staticmethod
    def non_local_means(image, h=10, template_window_size=7, search_window_size=21):
        """
        非局部均值滤波
        
        参数:
            image: 输入图像
            h: 滤波器强度参数
            template_window_size: 模板窗口大小
            search_window_size: 搜索窗口大小
            
        返回:
            滤波后的图像
        """
        if image is None:
            return None
            
        # 应用非局部均值滤波
        result = cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
        
        return result