#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何运算模块
实现图像的平移、旋转、缩放、镜像等几何变换
"""

import cv2
import numpy as np

class GeometricOperations:
    """几何运算类"""
    
    @staticmethod
    def translate(image, dx, dy):
        """
        图像平移
        
        参数:
            image: 输入图像
            dx: x方向平移量
            dy: y方向平移量
            
        返回:
            平移后的图像
        """
        if image is None:
            return None
            
        rows, cols = image.shape[:2]
        
        # 创建平移矩阵
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # 应用仿射变换
        result = cv2.warpAffine(image, M, (cols, rows))
        
        return result
    
    @staticmethod
    def rotate(image, angle, center=None, scale=1.0):
        """
        图像旋转
        
        参数:
            image: 输入图像
            angle: 旋转角度（正数表示逆时针旋转）
            center: 旋转中心，如果为None则使用图像中心
            scale: 缩放比例
            
        返回:
            旋转后的图像
        """
        if image is None:
            return None
            
        rows, cols = image.shape[:2]
        
        # 计算旋转中心
        if center is None:
            center = (cols // 2, rows // 2)
            
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 计算旋转后的图像尺寸
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_width = int((rows * sin) + (cols * cos))
        new_height = int((rows * cos) + (cols * sin))
        
        # 调整旋转矩阵的平移参数
        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]
        
        # 应用旋转
        result = cv2.warpAffine(image, M, (new_width, new_height))
        
        return result
    
    @staticmethod
    def scale(image, fx, fy=None, interpolation=cv2.INTER_LINEAR):
        """
        图像缩放
        
        参数:
            image: 输入图像
            fx: x方向的缩放比例
            fy: y方向的缩放比例，如果为None则使用fx
            interpolation: 插值方法
            
        返回:
            缩放后的图像
        """
        if image is None:
            return None
            
        if fy is None:
            fy = fx
            
        # 计算新的尺寸
        height, width = image.shape[:2]
        new_width = int(width * fx)
        new_height = int(height * fy)
        
        # 应用缩放
        result = cv2.resize(image, (new_width, new_height), 
                           interpolation=interpolation)
        
        return result
    
    @staticmethod
    def mirror(image, mode='horizontal'):
        """
        图像镜像
        
        参数:
            image: 输入图像
            mode: 镜像模式
                'horizontal': 水平镜像
                'vertical': 垂直镜像
                'both': 水平垂直镜像（旋转180度）
                
        返回:
            镜像后的图像
        """
        if image is None:
            return None
            
        if mode == 'horizontal':
            # 水平镜像
            return cv2.flip(image, 1)
        elif mode == 'vertical':
            # 垂直镜像
            return cv2.flip(image, 0)
        elif mode == 'both':
            # 水平垂直镜像
            return cv2.flip(image, -1)
        else:
            raise ValueError(f"不支持的镜像模式: {mode}")
    
    @staticmethod
    def perspective_transform(image, src_points, dst_points):
        """
        透视变换
        
        参数:
            image: 输入图像
            src_points: 源图像中的4个点
            dst_points: 目标图像中的4个点
            
        返回:
            透视变换后的图像
        """
        if image is None:
            return None
            
        # 确保点数为4
        if len(src_points) != 4 or len(dst_points) != 4:
            raise ValueError("需要4个点进行透视变换")
            
        # 转换为numpy数组
        src_pts = np.array(src_points, dtype=np.float32)
        dst_pts = np.array(dst_points, dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 计算输出图像尺寸
        height, width = image.shape[:2]
        
        # 应用透视变换
        result = cv2.warpPerspective(image, M, (width, height))
        
        return result
    
    @staticmethod
    def crop(image, x, y, width, height):
        """
        图像裁剪
        
        参数:
            image: 输入图像
            x: 裁剪区域左上角x坐标
            y: 裁剪区域左上角y坐标
            width: 裁剪区域宽度
            height: 裁剪区域高度
            
        返回:
            裁剪后的图像
        """
        if image is None:
            return None
            
        # 检查边界
        img_height, img_width = image.shape[:2]
        
        # 调整裁剪参数以确保在图像范围内
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = max(1, min(width, img_width - x))
        height = max(1, min(height, img_height - y))
        
        # 执行裁剪
        result = image[y:y+height, x:x+width]
        
        return result
    
    @staticmethod
    def resize_to_fit(image, target_width, target_height, keep_aspect_ratio=True):
        """
        调整图像尺寸以适应目标尺寸
        
        参数:
            image: 输入图像
            target_width: 目标宽度
            target_height: 目标高度
            keep_aspect_ratio: 是否保持宽高比
            
        返回:
            调整尺寸后的图像
        """
        if image is None:
            return None
            
        height, width = image.shape[:2]
        
        if keep_aspect_ratio:
            # 计算缩放比例
            scale_width = target_width / width
            scale_height = target_height / height
            scale = min(scale_width, scale_height)
            
            # 计算新尺寸
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 调整尺寸
            result = cv2.resize(image, (new_width, new_height))
            
            # 如果需要，在边缘填充
            if new_width < target_width or new_height < target_height:
                top = (target_height - new_height) // 2
                bottom = target_height - new_height - top
                left = (target_width - new_width) // 2
                right = target_width - new_width - left
                
                result = cv2.copyMakeBorder(result, top, bottom, left, right,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            # 直接调整尺寸
            result = cv2.resize(image, (target_width, target_height))
        
        return result