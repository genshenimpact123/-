#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分割模块
实现边缘检测、阈值分割、区域生长等图像分割算法
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import segmentation, filters

class ImageSegmentation:
    """图像分割类"""
    
    @staticmethod
    def edge_detection(image, method='canny', threshold1=100, threshold2=200, **kwargs):
        """
        边缘检测
        
        参数:
            image: 输入图像
            method: 边缘检测方法
                'canny': Canny边缘检测
                'sobel': Sobel边缘检测
                'prewitt': Prewitt边缘检测
                'laplacian': Laplacian边缘检测
                'log': LoG边缘检测
            threshold1: Canny算法的低阈值
            threshold2: Canny算法的高阈值
            **kwargs: 其他方法特定参数
            
        返回:
            边缘检测结果图像
        """
        if image is None:
            return None
            
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'canny':
            # Canny边缘检测
            return cv2.Canny(gray, threshold1, threshold2)
            
        elif method == 'sobel':
            # Sobel边缘检测
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kwargs.get('ksize', 3))
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kwargs.get('ksize', 3))
            
            # 计算梯度幅值
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # 归一化到0-255
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
            return magnitude.astype(np.uint8)
            
        elif method == 'prewitt':
            # Prewitt边缘检测
            kernelx = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])
            kernely = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]])
            
            prewittx = cv2.filter2D(gray, -1, kernelx)
            prewtty = cv2.filter2D(gray, -1, kernely)
            
            magnitude = np.sqrt(prewittx**2 + prewtty**2)
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
            return magnitude.astype(np.uint8)
            
        elif method == 'laplacian':
            # Laplacian边缘检测
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kwargs.get('ksize', 3))
            laplacian = np.abs(laplacian)
            laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
            
            return laplacian.astype(np.uint8)
            
        elif method == 'log':
            # LoG（Laplacian of Gaussian）边缘检测
            # 首先应用高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), kwargs.get('sigma', 1))
            # 然后应用Laplacian
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
            laplacian = np.abs(laplacian)
            laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
            
            return laplacian.astype(np.uint8)
            
        else:
            raise ValueError(f"不支持的边缘检测方法: {method}")
    
    @staticmethod
    def threshold_segmentation(image, method='otsu', threshold=127, **kwargs):
        """
        阈值分割
        
        参数:
            image: 输入图像
            method: 阈值分割方法
                'global': 全局阈值分割
                'otsu': Otsu阈值分割
                'adaptive': 自适应阈值分割
                'multi_otsu': 多级Otsu阈值分割
            threshold: 全局阈值（仅对'global'方法有效）
            **kwargs: 其他方法特定参数
            
        返回:
            二值化图像
        """
        if image is None:
            return None
            
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'global':
            # 全局阈值分割
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            return binary
            
        elif method == 'otsu':
            # Otsu阈值分割
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
            
        elif method == 'adaptive':
            # 自适应阈值分割
            block_size = kwargs.get('block_size', 11)
            c = kwargs.get('c', 2)
            
            binary = cv2.adaptiveThreshold(gray, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block_size, c)
            return binary
            
        elif method == 'multi_otsu':
            # 多级Otsu阈值分割
            from skimage.filters import threshold_multiotsu
            
            try:
                # 计算多级阈值
                thresholds = threshold_multiotsu(gray, classes=kwargs.get('classes', 3))
                
                # 应用多级阈值
                regions = np.digitize(gray, bins=thresholds)
                binary = np.uint8(regions * (255 // len(thresholds)))
                
                return binary
            except ImportError:
                # 如果skimage不可用，使用简单方法
                return ImageSegmentation.threshold_segmentation(gray, 'otsu')
            
        else:
            raise ValueError(f"不支持的阈值分割方法: {method}")
    
    @staticmethod
    def region_growing_segmentation(image, seed_point, threshold=10, connectivity=8):
        """
        区域生长法分割
        
        参数:
            image: 输入图像
            seed_point: 种子点坐标 (row, col)
            threshold: 生长阈值
            connectivity: 连接方式（4或8连通）
            
        返回:
            分割结果图像
        """
        if image is None:
            return None
            
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        rows, cols = gray.shape
        segmented = np.zeros_like(gray, dtype=np.uint8)
        
        # 检查种子点是否在图像范围内
        if not (0 <= seed_point[0] < rows and 0 <= seed_point[1] < cols):
            raise ValueError("种子点超出图像范围")
        
        # 种子点列表
        seed_list = [seed_point]
        segmented[seed_point] = 255
        
        # 种子点灰度值
        seed_value = gray[seed_point]
        
        # 定义邻域
        if connectivity == 4:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # 8连通
            neighbors = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
        
        while seed_list:
            current_point = seed_list.pop(0)
            
            for neighbor in neighbors:
                x = current_point[0] + neighbor[0]
                y = current_point[1] + neighbor[1]
                
                # 检查边界
                if 0 <= x < rows and 0 <= y < cols:
                    # 检查是否已访问且满足生长条件
                    if segmented[x, y] == 0:
                        if abs(int(gray[x, y]) - int(seed_value)) < threshold:
                            segmented[x, y] = 255
                            seed_list.append((x, y))
        
        return segmented
    
    @staticmethod
    def watershed_segmentation(image, markers=None):
        """
        分水岭算法分割
        
        参数:
            image: 输入图像
            markers: 标记图像，如果为None则自动生成
            
        返回:
            分割结果图像
        """
        if image is None:
            return None
            
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if markers is None:
            # 如果没有提供标记，使用Otsu阈值创建标记
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 形态学操作去除噪声
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # 确定背景区域
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # 确定前景区域
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            
            # 找到未知区域
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # 标记连通区域
            _, markers = cv2.connectedComponents(sure_fg)
            
            # 为分水岭算法添加标记
            markers = markers + 1
            markers[unknown == 255] = 0
        
        # 应用分水岭算法
        markers = cv2.watershed(image, markers)
        
        # 将标记转换为可视化图像
        result = np.zeros_like(gray, dtype=np.uint8)
        result[markers > 1] = 255  # 标记大于1的区域是分割出的物体
        
        return result
    
    @staticmethod
    def kmeans_segmentation(image, k=3, max_iter=100):
        """
        K-means聚类分割
        
        参数:
            image: 输入图像
            k: 聚类数量
            max_iter: 最大迭代次数
            
        返回:
            分割结果图像
        """
        if image is None:
            return None
            
        # 将图像转换为一维向量
        if len(image.shape) == 3:
            # 彩色图像
            vectorized = image.reshape((-1, 3))
        else:
            # 灰度图像
            vectorized = image.reshape((-1, 1))
        
        # 转换为浮点型
        vectorized = np.float32(vectorized)
        
        # 定义K-means算法的终止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
        
        # 应用K-means聚类
        _, labels, centers = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 将每个像素转换为其对应的中心颜色
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        
        # 重塑为原始图像形状
        if len(image.shape) == 3:
            result = segmented.reshape((image.shape))
        else:
            result = segmented.reshape((image.shape[0], image.shape[1]))
        
        return result
    
    @staticmethod
    def grabcut_segmentation(image, rect=None, iter_count=5):
        """
        GrabCut算法分割
        
        参数:
            image: 输入图像
            rect: 包含前景的矩形区域 (x, y, w, h)，如果为None则需要手动标记
            iter_count: 迭代次数
            
        返回:
            分割结果图像
        """
        if image is None:
            return None
            
        # 创建掩膜
        mask = np.zeros(image.shape[:2], np.uint8)
        
        # 创建背景和前景模型
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        if rect is None:
            # 如果没有提供矩形，使用全图作为可能的前景
            rect = (0, 0, image.shape[1], image.shape[0])
        
        # 应用GrabCut算法
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
        
        # 创建结果掩膜
        result_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 应用掩膜
        result = image * result_mask[:, :, np.newaxis]
        
        return result
    
    @staticmethod
    def active_contour_segmentation(image, init_snake=None, alpha=0.01, beta=0.1, gamma=0.01, 
                                   max_iterations=1000, convergence=0.1):
        """
        主动轮廓模型（Snake）分割
        
        参数:
            image: 输入图像
            init_snake: 初始轮廓点，如果为None则使用圆形轮廓
            alpha: 弹性力权重
            beta: 弯曲力权重
            gamma: 时间步长
            max_iterations: 最大迭代次数
            convergence: 收敛阈值
            
        返回:
            带有轮廓的图像
        """
        if image is None:
            return None
            
        # 确保图像是灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 计算图像梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        # 计算梯度幅值作为外部能量
        external_energy = -(sobelx**2 + sobely**2)
        external_energy = cv2.normalize(external_energy, None, 0, 1, cv2.NORM_MINMAX)
        
        rows, cols = gray.shape
        
        # 如果没有提供初始轮廓，使用圆形轮廓
        if init_snake is None:
            center_x, center_y = cols // 2, rows // 2
            radius = min(rows, cols) // 4
            theta = np.linspace(0, 2 * np.pi, 50)
            init_snake = np.array([
                center_x + radius * np.cos(theta),
                center_y + radius * np.sin(theta)
            ]).T
        
        snake = init_snake.copy()
        
        # 构建蛇模型的内部能量矩阵（五对角矩阵）
        n = len(snake)
        eye_n = np.eye(n, dtype=float)
        
        # 创建A矩阵
        a = np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) - 2 * eye_n
        b = np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1) - 4 * np.roll(eye_n, -1, axis=0) - \
            4 * np.roll(eye_n, -1, axis=1) + 6 * eye_n
        
        A = -alpha * a + beta * b
        
        # 计算(A + gamma * I)的逆
        inv = np.linalg.inv(A + gamma * eye_n)
        
        # 迭代优化
        for i in range(max_iterations):
            # 计算外部力
            fx = ndimage.map_coordinates(external_energy, [snake[:, 1], snake[:, 0]], order=1, mode='nearest')
            
            # 更新蛇的位置
            snake = np.dot(inv, gamma * snake + fx[:, np.newaxis] * np.array([1, 1]))
            
            # 检查收敛
            if i > 0 and np.linalg.norm(snake - prev_snake) < convergence:
                break
                
            prev_snake = snake.copy()
        
        # 创建结果图像
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 绘制轮廓
        snake_int = snake.astype(np.int32)
        for i in range(len(snake_int)):
            cv2.circle(result, (snake_int[i, 0], snake_int[i, 1]), 2, (0, 255, 0), -1)
            if i > 0:
                cv2.line(result, (snake_int[i-1, 0], snake_int[i-1, 1]), 
                        (snake_int[i, 0], snake_int[i, 1]), (0, 255, 0), 1)
        
        # 连接首尾点
        cv2.line(result, (snake_int[-1, 0], snake_int[-1, 1]), 
                (snake_int[0, 0], snake_int[0, 1]), (0, 255, 0), 1)
        
        return result