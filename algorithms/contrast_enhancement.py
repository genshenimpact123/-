#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比度增强模块
实现图像的灰度变换和直方图均衡化等对比度增强算法
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

class ContrastEnhancement:
    """对比度增强类"""
    
    @staticmethod
    def grayscale_transform(image, method='linear', alpha=1.0, beta=0, gamma=1.0):
        """
        灰度变换
        
        参数:
            image: 输入图像
            method: 变换方法
                'linear': 线性变换
                'gamma': Gamma校正
                'log': 对数变换
                'exp': 指数变换
            alpha: 线性变换的斜率（仅对线性变换有效）
            beta: 线性变换的截距（仅对线性变换有效）
            gamma: Gamma值（仅对Gamma校正有效）
                
        返回:
            变换后的图像
        """
        if image is None:
            return None
            
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'linear':
            # 线性变换：g(x,y) = α * f(x,y) + β
            result = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
        elif method == 'gamma':
            # Gamma校正：g(x,y) = c * [f(x,y)]^γ
            # 构建Gamma查找表
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            
            # 应用Gamma校正
            result = cv2.LUT(gray, table)
            
        elif method == 'log':
            # 对数变换：g(x,y) = c * log(1 + f(x,y))
            # 将图像转换为浮点数
            c = 255 / np.log(1 + np.max(gray))
            log_image = c * (np.log(gray.astype(np.float32) + 1))
            result = np.array(log_image, dtype=np.uint8)
            
        elif method == 'exp':
            # 指数变换：g(x,y) = c * [f(x,y)]^γ
            # 这里使用一个简化的指数变换
            result = np.array(255 * (gray / 255.0) ** gamma, dtype=np.uint8)
            
        else:
            raise ValueError(f"不支持的变换方法: {method}")
        
        return result
    
    @staticmethod
    def histogram_equalization(image):
        """
        直方图均衡化
        
        参数:
            image: 输入图像
            
        返回:
            均衡化后的图像
        """
        if image is None:
            return None
            
        if len(image.shape) == 2:
            # 灰度图像
            return cv2.equalizeHist(image)
        else:
            # 彩色图像 - 对每个通道分别处理
            # 转换为YUV色彩空间，只对Y通道进行均衡化
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return result
    
    @staticmethod
    def adaptive_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
        """
        自适应直方图均衡化（CLAHE）
        
        参数:
            image: 输入图像
            clip_limit: 对比度限制阈值
            grid_size: 网格大小
            
        返回:
            自适应均衡化后的图像
        """
        if image is None:
            return None
            
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        
        if len(image.shape) == 2:
            # 灰度图像
            return clahe.apply(image)
        else:
            # 彩色图像 - 对每个通道分别处理
            # 转换为LAB色彩空间，只对L通道进行均衡化
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return result
    
    @staticmethod
    def contrast_stretching(image, low_percent=2, high_percent=98):
        """
        对比度拉伸
        
        参数:
            image: 输入图像
            low_percent: 低百分比阈值
            high_percent: 高百分比阈值
            
        返回:
            对比度拉伸后的图像
        """
        if image is None:
            return None
            
        if len(image.shape) == 3:
            # 彩色图像转换为灰度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 计算百分比阈值
        low_val = np.percentile(gray, low_percent)
        high_val = np.percentile(gray, high_percent)
        
        # 线性拉伸
        stretched = np.clip((gray - low_val) * (255.0 / (high_val - low_val)), 0, 255)
        
        return stretched.astype(np.uint8)
    
    @staticmethod
    def histogram_matching(source_image, reference_image):
        """
        直方图匹配（规定化）
        
        参数:
            source_image: 源图像
            reference_image: 参考图像
            
        返回:
            直方图匹配后的图像
        """
        if source_image is None or reference_image is None:
            return None
            
        # 确保图像是灰度图
        if len(source_image.shape) == 3:
            source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source_image.copy()
            
        if len(reference_image.shape) == 3:
            reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            reference_gray = reference_image.copy()
        
        # 计算直方图
        src_hist, _ = np.histogram(source_gray.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference_gray.flatten(), 256, [0, 256])
        
        # 计算累积分布函数
        src_cdf = src_hist.cumsum()
        src_cdf_normalized = src_cdf / src_cdf[-1]
        
        ref_cdf = ref_hist.cumsum()
        ref_cdf_normalized = ref_cdf / ref_cdf[-1]
        
        # 创建映射表
        mapping = np.zeros(256, dtype=np.uint8)
        
        for i in range(256):
            # 找到最接近的参考CDF值
            j = 255
            while j >= 0 and ref_cdf_normalized[j] >= src_cdf_normalized[i]:
                j -= 1
            mapping[i] = max(0, j + 1)
        
        # 应用映射
        result = cv2.LUT(source_gray, mapping)
        
        return result
    
    @staticmethod
    def retinex_enhancement(image, sigma_list=(15, 80, 250)):
        """
        Retinex图像增强算法
        
        参数:
            image: 输入图像
            sigma_list: 高斯核尺度列表
            
        返回:
            增强后的图像
        """
        if image is None:
            return None
            
        # 将图像转换为浮点型
        img_float = image.astype(np.float32) / 255.0
        
        if len(img_float.shape) == 2:
            # 灰度图像
            gray = img_float
            result = np.zeros_like(gray)
            
            for sigma in sigma_list:
                # 高斯模糊
                blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
                # 计算反射分量
                reflection = np.log(gray + 1e-6) - np.log(blurred + 1e-6)
                result += reflection
                
            # 平均反射分量
            result = result / len(sigma_list)
            
            # 归一化
            result = (result - result.min()) / (result.max() - result.min())
            result = (result * 255).astype(np.uint8)
            
        else:
            # 彩色图像
            result = np.zeros_like(img_float)
            
            for channel in range(3):
                channel_img = img_float[:, :, channel]
                channel_result = np.zeros_like(channel_img)
                
                for sigma in sigma_list:
                    blurred = cv2.GaussianBlur(channel_img, (0, 0), sigma)
                    reflection = np.log(channel_img + 1e-6) - np.log(blurred + 1e-6)
                    channel_result += reflection
                
                channel_result = channel_result / len(sigma_list)
                result[:, :, channel] = channel_result
            
            # 归一化
            for channel in range(3):
                channel_min = result[:, :, channel].min()
                channel_max = result[:, :, channel].max()
                if channel_max > channel_min:
                    result[:, :, channel] = (result[:, :, channel] - channel_min) / (channel_max - channel_min)
            
            result = (result * 255).astype(np.uint8)
        
        return result