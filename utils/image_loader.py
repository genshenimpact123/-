#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像加载工具模块
提供图像加载、格式转换、基本信息获取等功能
"""

import cv2
import numpy as np
from PIL import Image
import os
import magic  # 替代imghdr

class ImageLoader:
    """图像加载工具类"""
    
    @staticmethod
    def load_image(file_path, mode='color'):
        """
        加载图像文件
        
        参数:
            file_path: 图像文件路径
            mode: 加载模式
                'color': 加载彩色图像（默认）
                'grayscale': 加载灰度图像
                'unchanged': 按原样加载（包括alpha通道）
                
        返回:
            加载的图像（numpy数组）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        # 检查文件是否为图像 - 使用文件扩展名
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in image_extensions:
            # 尝试使用magic库检测文件类型
            try:
                import magic
                mime = magic.Magic(mime=True)
                file_type = mime.from_file(file_path)
                if not file_type.startswith('image/'):
                    raise ValueError(f"文件不是有效的图像格式: {file_path}")
            except ImportError:
                # 如果magic库不可用，仅检查扩展名
                raise ValueError(f"文件扩展名不是支持的图像格式: {ext}")
        
        # 根据模式选择加载标志
        if mode == 'grayscale':
            flags = cv2.IMREAD_GRAYSCALE
        elif mode == 'unchanged':
            flags = cv2.IMREAD_UNCHANGED
        else:  # 'color'
            flags = cv2.IMREAD_COLOR
        
        # 加载图像
        image = cv2.imread(file_path, flags)
        
        if image is None:
            # 尝试使用PIL作为后备方案
            try:
                pil_image = Image.open(file_path)
                image = np.array(pil_image)
                # 如果加载为RGB但需要BGR，进行转换
                if mode == 'color' and len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"无法加载图像文件: {file_path} - {str(e)}")
        
        # 如果是彩色图像，将BGR转换为RGB
        if len(image.shape) == 3 and mode == 'color':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    
    @staticmethod
    def load_image_pil(file_path):
        """
        使用PIL加载图像
        
        参数:
            file_path: 图像文件路径
            
        返回:
            PIL Image对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            image = Image.open(file_path)
            return image
        except Exception as e:
            raise ValueError(f"无法加载图像文件: {file_path} - {str(e)}")
    
    @staticmethod
    def save_image(image, file_path, quality=95):
        """
        保存图像文件
        
        参数:
            image: 要保存的图像（numpy数组）
            file_path: 保存路径
            quality: JPEG质量（0-100）
        """
        if image is None:
            raise ValueError("图像不能为空")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 根据扩展名选择保存方式
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg']:
            # 对于JPEG，需要将RGB转换为BGR
            if len(image.shape) == 3:
                save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                save_image = image
            
            cv2.imwrite(file_path, save_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        elif ext == '.png':
            # PNG格式，可以包含alpha通道
            if len(image.shape) == 3:
                save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                save_image = image
            
            cv2.imwrite(file_path, save_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        else:
            # 其他格式
            if len(image.shape) == 3:
                save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                save_image = image
            
            cv2.imwrite(file_path, save_image)
    
    @staticmethod
    def convert_to_grayscale(image):
        """
        将图像转换为灰度图
        
        参数:
            image: 输入图像
            
        返回:
            灰度图像
        """
        if image is None:
            return None
            
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            return image.copy()
    
    @staticmethod
    def convert_to_rgb(image):
        """
        将图像转换为RGB格式
        
        参数:
            image: 输入图像
            
        返回:
            RGB图像
        """
        if image is None:
            return None
            
        if len(image.shape) == 2:
            # 灰度转RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA转RGB
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            # 检查是否为BGR
            # 简单检查：如果第一个像素的B>R，可能是BGR
            if image[0, 0, 0] > image[0, 0, 2]:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return image.copy()
        else:
            return image.copy()
    
    @staticmethod
    def get_image_info(image):
        """
        获取图像信息
        
        参数:
            image: 输入图像
            
        返回:
            包含图像信息的字典
        """
        if image is None:
            return None
            
        info = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'std': float(np.std(image))
        }
        
        if len(image.shape) == 3:
            info['channels'] = image.shape[2]
            info['type'] = 'color'
            
            # 计算每个通道的统计信息
            channel_info = {}
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                channel_info[f'channel_{i}'] = {
                    'min': float(np.min(channel)),
                    'max': float(np.max(channel)),
                    'mean': float(np.mean(channel)),
                    'std': float(np.std(channel))
                }
            info['channel_stats'] = channel_info
        else:
            info['channels'] = 1
            info['type'] = 'grayscale'
        
        return info
    
    @staticmethod
    def resize_image(image, width=None, height=None, keep_aspect_ratio=True):
        """
        调整图像尺寸
        
        参数:
            image: 输入图像
            width: 目标宽度
            height: 目标高度
            keep_aspect_ratio: 是否保持宽高比
            
        返回:
            调整尺寸后的图像
        """
        if image is None:
            return None
            
        h, w = image.shape[:2]
        
        if width is None and height is None:
            return image.copy()
        
        if keep_aspect_ratio:
            if width is not None and height is not None:
                # 计算缩放比例
                scale_width = width / w
                scale_height = height / h
                scale = min(scale_width, scale_height)
                
                new_width = int(w * scale)
                new_height = int(h * scale)
            elif width is not None:
                scale = width / w
                new_width = width
                new_height = int(h * scale)
            else:  # height is not None
                scale = height / h
                new_width = int(w * scale)
                new_height = height
        else:
            new_width = width if width is not None else w
            new_height = height if height is not None else h
        
        # 调整尺寸
        resized = cv2.resize(image, (new_width, new_height))
        
        return resized
    
    @staticmethod
    def crop_image(image, x, y, width, height):
        """
        裁剪图像
        
        参数:
            image: 输入图像
            x: 左上角x坐标
            y: 左上角y坐标
            width: 裁剪宽度
            height: 裁剪高度
            
        返回:
            裁剪后的图像
        """
        if image is None:
            return None
            
        h, w = image.shape[:2]
        
        # 确保裁剪区域在图像范围内
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = max(1, min(width, w - x))
        height = max(1, min(height, h - y))
        
        cropped = image[y:y+height, x:x+width]
        
        return cropped
    
    @staticmethod
    def rotate_image(image, angle, center=None, scale=1.0):
        """
        旋转图像
        
        参数:
            image: 输入图像
            angle: 旋转角度（正数表示逆时针）
            center: 旋转中心
            scale: 缩放比例
            
        返回:
            旋转后的图像
        """
        if image is None:
            return None
            
        h, w = image.shape[:2]
        
        if center is None:
            center = (w // 2, h // 2)
        
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 计算旋转后的图像尺寸
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 调整旋转矩阵的平移参数
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # 旋转图像
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        
        return rotated
    
    @staticmethod
    def normalize_image(image, min_val=0, max_val=255):
        """
        归一化图像
        
        参数:
            image: 输入图像
            min_val: 归一化后的最小值
            max_val: 归一化后的最大值
            
        返回:
            归一化后的图像
        """
        if image is None:
            return None
            
        # 计算当前图像的取值范围
        current_min = np.min(image)
        current_max = np.max(image)
        
        if current_max == current_min:
            # 如果所有像素值相同
            normalized = np.full_like(image, min_val, dtype=np.float32)
        else:
            # 线性归一化
            normalized = (image - current_min) / (current_max - current_min)
            normalized = normalized * (max_val - min_val) + min_val
        
        return normalized.astype(image.dtype)
    
    @staticmethod
    def create_thumbnail(image, max_size=200):
        """
        创建缩略图
        
        参数:
            image: 输入图像
            max_size: 缩略图的最大尺寸（宽度或高度）
            
        返回:
            缩略图
        """
        if image is None:
            return None
            
        h, w = image.shape[:2]
        
        if h > w:
            # 竖图
            new_height = max_size
            new_width = int(w * (max_size / h))
        else:
            # 横图或正方形
            new_width = max_size
            new_height = int(h * (max_size / w))
        
        thumbnail = cv2.resize(image, (new_width, new_height))
        
        return thumbnail
    
    @staticmethod
    def get_dominant_colors(image, k=5):
        """
        获取图像的主要颜色
        
        参数:
            image: 输入图像
            k: 颜色数量
            
        返回:
            主要颜色列表（RGB格式）
        """
        if image is None:
            return []
            
        # 确保图像是彩色图像
        if len(image.shape) == 2:
            # 如果是灰度图，转换为伪彩色
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # 如果有alpha通道，去除它
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # 将图像转换为一维向量
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # 定义K-means算法的终止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # 应用K-means聚类
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 转换为整数
        centers = np.uint8(centers)
        
        # 按颜色频率排序
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)  # 降序排列
        
        dominant_colors = []
        for idx in sorted_indices:
            color = centers[unique_labels[idx]].tolist()
            percentage = (counts[idx] / len(labels)) * 100
            dominant_colors.append({
                'color': color,
                'percentage': percentage
            })
        
        return dominant_colors