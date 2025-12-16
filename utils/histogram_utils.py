#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直方图工具模块
提供图像直方图计算、显示和分析功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class HistogramUtils:
    """直方图工具类"""
    
    @staticmethod
    def compute_histogram(image, bins=256, range=(0, 256)):
        """
        计算图像直方图
        
        参数:
            image: 输入图像
            bins: 直方图箱子数量
            range: 像素值范围
            
        返回:
            直方图数据
        """
        if image is None:
            return None
            
        if len(image.shape) == 3:
            # 彩色图像 - 分别计算每个通道的直方图
            histograms = []
            colors = ('b', 'g', 'r')
            
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [bins], range)
                histograms.append(hist)
            
            return histograms
        else:
            # 灰度图像
            hist = cv2.calcHist([image], [0], None, [bins], range)
            return hist
    
    @staticmethod
    def compute_cumulative_histogram(histogram):
        """
        计算累积直方图
        
        参数:
            histogram: 输入直方图
            
        返回:
            累积直方图
        """
        if histogram is None:
            return None
            
        if isinstance(histogram, list):
            # 彩色图像的直方图列表
            cumulative_hists = []
            for hist in histogram:
                cumulative_hist = hist.cumsum()
                cumulative_hist_normalized = cumulative_hist / cumulative_hist[-1]
                cumulative_hists.append(cumulative_hist_normalized)
            return cumulative_hists
        else:
            # 灰度图像的直方图
            cumulative_hist = histogram.cumsum()
            cumulative_hist_normalized = cumulative_hist / cumulative_hist[-1]
            return cumulative_hist_normalized
    
    @staticmethod
    def plot_histogram(image, ax=None, title=None, show_cumulative=False):
        """
        绘制直方图
        
        参数:
            image: 输入图像
            ax: matplotlib轴对象，如果为None则创建新的图形
            title: 图形标题
            show_cumulative: 是否显示累积直方图
            
        返回:
            matplotlib图形对象
        """
        if image is None:
            return None
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            new_figure = True
        else:
            new_figure = False
        
        if len(image.shape) == 3:
            # 彩色图像
            colors = ('b', 'g', 'r')
            color_names = ('Blue', 'Green', 'Red')
            
            for i, (color, name) in enumerate(zip(colors, color_names)):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                ax.plot(hist, color=color, label=name, linewidth=1.5)
                
                if show_cumulative:
                    cumulative_hist = hist.cumsum()
                    cumulative_hist_normalized = cumulative_hist / cumulative_hist[-1] * hist.max()
                    ax.plot(cumulative_hist_normalized, color=color, linestyle='--', 
                           alpha=0.7, label=f'{name} Cumulative')
            
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.set_title(title or 'Color Histogram')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            # 灰度图像
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            ax.plot(hist, color='black', linewidth=2, label='Intensity')
            
            if show_cumulative:
                cumulative_hist = hist.cumsum()
                cumulative_hist_normalized = cumulative_hist / cumulative_hist[-1] * hist.max()
                ax.plot(cumulative_hist_normalized, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.7, label='Cumulative')
            
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.set_title(title or 'Grayscale Histogram')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        if new_figure:
            plt.tight_layout()
            return fig
        else:
            return ax
    
    @staticmethod
    def plot_histogram_2d(image1, image2, ax=None, title=None):
        """
        绘制二维直方图（用于比较两个图像）
        
        参数:
            image1: 第一个图像
            image2: 第二个图像
            ax: matplotlib轴对象
            title: 图形标题
            
        返回:
            matplotlib图形对象
        """
        if image1 is None or image2 is None:
            return None
            
        # 确保两个图像尺寸相同
        if image1.shape != image2.shape:
            # 调整第二个图像的尺寸
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # 转换为灰度图
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = image1
            gray2 = image2
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            new_figure = True
        else:
            new_figure = False
        
        # 计算二维直方图
        hist_2d, xedges, yedges = np.histogram2d(
            gray1.flatten(), gray2.flatten(), bins=50, range=[[0, 256], [0, 256]]
        )
        
        # 绘制二维直方图
        im = ax.imshow(np.rot90(hist_2d), cmap=plt.cm.hot,
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        ax.set_xlabel('Image 1 Pixel Intensity')
        ax.set_ylabel('Image 2 Pixel Intensity')
        ax.set_title(title or '2D Histogram Comparison')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Frequency')
        
        # 添加对角线（理想的相关线）
        ax.plot([0, 256], [0, 256], 'b--', alpha=0.5, linewidth=1)
        
        if new_figure:
            plt.tight_layout()
            return fig
        else:
            return ax
    
    @staticmethod
    def get_histogram_statistics(histogram):
        """
        获取直方图统计信息
        
        参数:
            histogram: 输入直方图
            
        返回:
            统计信息字典
        """
        if histogram is None:
            return None
            
        if isinstance(histogram, list):
            # 彩色图像
            stats = {}
            channel_names = ['Blue', 'Green', 'Red']
            
            for i, (hist, name) in enumerate(zip(histogram, channel_names)):
                stats[name] = HistogramUtils._compute_channel_stats(hist)
            
            return stats
        else:
            # 灰度图像
            return HistogramUtils._compute_channel_stats(histogram)
    
    @staticmethod
    def _compute_channel_stats(histogram):
        """计算单个通道的统计信息"""
        # 计算总像素数
        total_pixels = np.sum(histogram)
        
        if total_pixels == 0:
            return {}
        
        # 计算均值
        intensity_values = np.arange(len(histogram))
        mean = np.sum(intensity_values * histogram.flatten()) / total_pixels
        
        # 计算方差和标准差
        variance = np.sum(((intensity_values - mean) ** 2) * histogram.flatten()) / total_pixels
        std_dev = np.sqrt(variance)
        
        # 计算中位数
        cumulative_sum = np.cumsum(histogram)
        median_index = np.where(cumulative_sum >= total_pixels / 2)[0][0]
        
        # 计算众数（出现频率最高的强度值）
        mode_index = np.argmax(histogram)
        
        # 计算偏度和峰度
        skewness = (np.sum(((intensity_values - mean) ** 3) * histogram.flatten()) / 
                   total_pixels) / (std_dev ** 3)
        kurtosis = (np.sum(((intensity_values - mean) ** 4) * histogram.flatten()) / 
                   total_pixels) / (std_dev ** 4) - 3
        
        # 计算熵
        probabilities = histogram.flatten() / total_pixels
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return {
            'mean': float(mean),
            'std_dev': float(std_dev),
            'median': int(median_index),
            'mode': int(mode_index),
            'variance': float(variance),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'entropy': float(entropy),
            'total_pixels': int(total_pixels)
        }
    
    @staticmethod
    def equalize_histogram(image):
        """
        直方图均衡化
        
        参数:
            image: 输入图像
            
        返回:
            均衡化后的图像
        """
        if image is None:
            return None
            
        if len(image.shape) == 3:
            # 彩色图像 - 转换为YUV色彩空间，对Y通道进行均衡化
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            # 灰度图像
            equalized = cv2.equalizeHist(image)
        
        return equalized
    
    @staticmethod
    def match_histograms(source_image, reference_image):
        """
        直方图匹配（规定化）
        
        参数:
            source_image: 源图像
            reference_image: 参考图像
            
        返回:
            匹配后的图像
        """
        if source_image is None or reference_image is None:
            return None
            
        # 确保图像是灰度图
        if len(source_image.shape) == 3:
            source_gray = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)
        else:
            source_gray = source_image.copy()
            
        if len(reference_image.shape) == 3:
            reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
        else:
            reference_gray = reference_image.copy()
        
        # 调整参考图像尺寸以匹配源图像
        if reference_gray.shape != source_gray.shape:
            reference_gray = cv2.resize(reference_gray, 
                                       (source_gray.shape[1], source_gray.shape[0]))
        
        # 计算直方图
        source_hist, _ = np.histogram(source_gray.flatten(), 256, [0, 256])
        reference_hist, _ = np.histogram(reference_gray.flatten(), 256, [0, 256])
        
        # 计算累积分布函数
        source_cdf = source_hist.cumsum()
        source_cdf_normalized = source_cdf / source_cdf[-1]
        
        reference_cdf = reference_hist.cumsum()
        reference_cdf_normalized = reference_cdf / reference_cdf[-1]
        
        # 创建映射表
        mapping = np.zeros(256, dtype=np.uint8)
        
        for i in range(256):
            # 找到最接近的参考CDF值
            j = 255
            while j >= 0 and reference_cdf_normalized[j] >= source_cdf_normalized[i]:
                j -= 1
            mapping[i] = max(0, j + 1)
        
        # 应用映射
        matched = cv2.LUT(source_gray, mapping)
        
        return matched
    
    @staticmethod
    def compute_histogram_distance(hist1, hist2, method='correlation'):
        """
        计算两个直方图之间的距离
        
        参数:
            hist1: 第一个直方图
            hist2: 第二个直方图
            method: 距离计算方法
                'correlation': 相关系数
                'chi_square': 卡方距离
                'intersection': 交集距离
                'hellinger': Hellinger距离
                'bhattacharyya': Bhattacharyya距离
                
        返回:
            距离值
        """
        if hist1 is None or hist2 is None:
            return None
            
        # 确保直方图形状相同
        if hist1.shape != hist2.shape:
            # 调整第二个直方图
            hist2 = cv2.resize(hist2, hist1.shape)
        
        # 归一化直方图
        hist1_norm = cv2.normalize(hist1, None, norm_type=cv2.NORM_L1).flatten()
        hist2_norm = cv2.normalize(hist2, None, norm_type=cv2.NORM_L1).flatten()
        
        if method == 'correlation':
            # 相关系数（1表示完全相关，-1表示完全负相关）
            return cv2.compareHist(hist1_norm, hist2_norm, cv2.HISTCMP_CORREL)
            
        elif method == 'chi_square':
            # 卡方距离（0表示完全匹配）
            return cv2.compareHist(hist1_norm, hist2_norm, cv2.HISTCMP_CHISQR)
            
        elif method == 'intersection':
            # 交集距离（1表示完全匹配）
            return cv2.compareHist(hist1_norm, hist2_norm, cv2.HISTCMP_INTERSECT)
            
        elif method == 'hellinger':
            # Hellinger距离（0表示完全匹配）
            return cv2.compareHist(hist1_norm, hist2_norm, cv2.HISTCMP_HELLINGER)
            
        elif method == 'bhattacharyya':
            # Bhattacharyya距离（0表示完全匹配）
            return cv2.compareHist(hist1_norm, hist2_norm, cv2.HISTCMP_BHATTACHARYYA)
            
        else:
            raise ValueError(f"不支持的距离计算方法: {method}")
    
    @staticmethod
    def create_histogram_image(histogram, width=400, height=300, title="Histogram"):
        """
        创建直方图图像
        
        参数:
            histogram: 直方图数据
            width: 图像宽度
            height: 图像高度
            title: 图像标题
            
        返回:
            直方图图像（numpy数组）
        """
        if histogram is None:
            return None
            
        # 创建图形
        fig = Figure(figsize=(width/100, height/100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        if isinstance(histogram, list):
            # 彩色图像直方图
            colors = ('b', 'g', 'r')
            color_names = ('Blue', 'Green', 'Red')
            
            for i, (hist, color, name) in enumerate(zip(histogram, colors, color_names)):
                ax.plot(hist, color=color, label=name, linewidth=1.5)
            
            ax.legend()
        else:
            # 灰度图像直方图
            ax.plot(histogram, color='black', linewidth=2)
        
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # 将图形转换为numpy数组
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        
        return image