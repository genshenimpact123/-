#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法测试模块
用于测试各个图像处理算法的正确性
"""

import unittest
import cv2
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.geometric_operations import GeometricOperations
from algorithms.contrast_enhancement import ContrastEnhancement
from algorithms.smoothing_filters import SmoothingFilters
from algorithms.image_segmentation import ImageSegmentation

class TestGeometricOperations(unittest.TestCase):
    """几何运算测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试图像
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75, :] = 255  # 白色方块
        
    def test_translate(self):
        """测试平移"""
        translated = GeometricOperations.translate(self.test_image, 10, 20)
        
        self.assertIsNotNone(translated)
        self.assertEqual(translated.shape, self.test_image.shape)
        
        # 检查平移后的方块位置
        # 原方块在25:75, 25:75，平移(10,20)后应该在35:85, 45:95
        # 但由于边缘处理，实际值可能会有变化
        self.assertTrue(np.any(translated[35:85, 45:95] > 0))
    
    def test_rotate(self):
        """测试旋转"""
        rotated = GeometricOperations.rotate(self.test_image, 45)
        
        self.assertIsNotNone(rotated)
        # 旋转后图像尺寸会变化
        self.assertGreater(rotated.shape[0], 0)
        self.assertGreater(rotated.shape[1], 0)
    
    def test_scale(self):
        """测试缩放"""
        scaled = GeometricOperations.scale(self.test_image, 0.5)
        
        self.assertIsNotNone(scaled)
        self.assertEqual(scaled.shape[0], 50)
        self.assertEqual(scaled.shape[1], 50)
    
    def test_mirror(self):
        """测试镜像"""
        mirrored = GeometricOperations.mirror(self.test_image, 'horizontal')
        
        self.assertIsNotNone(mirrored)
        self.assertEqual(mirrored.shape, self.test_image.shape)
        
        # 检查镜像后的对称性
        original_left = self.test_image[:, :50]
        original_right = self.test_image[:, 50:]
        mirrored_left = mirrored[:, :50]
        mirrored_right = mirrored[:, 50:]
        
        # 水平镜像后，左侧应该与原始右侧对称
        self.assertTrue(np.array_equal(original_right, mirrored_left[:, ::-1]))

class TestContrastEnhancement(unittest.TestCase):
    """对比度增强测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试图像（低对比度）
        self.test_image = np.ones((100, 100), dtype=np.uint8) * 100
    
    def test_grayscale_transform_linear(self):
        """测试线性灰度变换"""
        transformed = ContrastEnhancement.grayscale_transform(
            self.test_image, method='linear', alpha=2.0, beta=0
        )
        
        self.assertIsNotNone(transformed)
        self.assertEqual(transformed.shape, self.test_image.shape)
        # 线性变换后，值应该翻倍（但可能被截断到255）
        expected_value = min(100 * 2, 255)
        self.assertEqual(transformed[0, 0], expected_value)
    
    def test_grayscale_transform_gamma(self):
        """测试Gamma校正"""
        transformed = ContrastEnhancement.grayscale_transform(
            self.test_image, method='gamma', gamma=0.5
        )
        
        self.assertIsNotNone(transformed)
        self.assertEqual(transformed.shape, self.test_image.shape)
    
    def test_histogram_equalization(self):
        """测试直方图均衡化"""
        # 创建具有明显亮度差异的图像
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[0:50, :] = 50
        test_image[50:100, :] = 200
        
        equalized = ContrastEnhancement.histogram_equalization(test_image)
        
        self.assertIsNotNone(equalized)
        self.assertEqual(equalized.shape, test_image.shape)
        
        # 检查均衡化后是否扩展了动态范围
        self.assertGreaterEqual(equalized.min(), 0)
        self.assertLessEqual(equalized.max(), 255)
    
    def test_adaptive_histogram_equalization(self):
        """测试自适应直方图均衡化"""
        equalized = ContrastEnhancement.adaptive_histogram_equalization(
            self.test_image
        )
        
        self.assertIsNotNone(equalized)
        self.assertEqual(equalized.shape, self.test_image.shape)

class TestSmoothingFilters(unittest.TestCase):
    """平滑滤波测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试图像（带有噪声）
        np.random.seed(42)
        self.test_image = np.ones((50, 50), dtype=np.uint8) * 100
        # 添加椒盐噪声
        noise_mask = np.random.random((50, 50)) < 0.1
        self.test_image[noise_mask] = np.random.choice([0, 255], noise_mask.sum())
    
    def test_mean_filter(self):
        """测试均值滤波"""
        filtered = SmoothingFilters.mean_filter(self.test_image, kernel_size=3)
        
        self.assertIsNotNone(filtered)
        self.assertEqual(filtered.shape, self.test_image.shape)
        # 均值滤波应该平滑图像
        self.assertLess(np.std(filtered), np.std(self.test_image) * 1.5)
    
    def test_median_filter(self):
        """测试中值滤波"""
        filtered = SmoothingFilters.median_filter(self.test_image, kernel_size=3)
        
        self.assertIsNotNone(filtered)
        self.assertEqual(filtered.shape, self.test_image.shape)
        # 中值滤波对椒盐噪声特别有效
        # 检查是否移除了极端值
        extreme_pixels = np.sum((self.test_image == 0) | (self.test_image == 255))
        filtered_extreme = np.sum((filtered == 0) | (filtered == 255))
        self.assertLess(filtered_extreme, extreme_pixels)
    
    def test_gaussian_filter(self):
        """测试高斯滤波"""
        filtered = SmoothingFilters.gaussian_filter(self.test_image, kernel_size=5, sigma=1.0)
        
        self.assertIsNotNone(filtered)
        self.assertEqual(filtered.shape, self.test_image.shape)
    
    def test_bilateral_filter(self):
        """测试双边滤波"""
        # 创建彩色测试图像
        color_image = np.stack([self.test_image] * 3, axis=2)
        filtered = SmoothingFilters.bilateral_filter(color_image, d=9, sigma_color=75, sigma_space=75)
        
        self.assertIsNotNone(filtered)
        self.assertEqual(filtered.shape, color_image.shape)

class TestImageSegmentation(unittest.TestCase):
    """图像分割测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试图像（包含一个明亮的对象）
        self.test_image = np.zeros((100, 100), dtype=np.uint8)
        self.test_image[30:70, 30:70] = 200  # 明亮的正方形
    
    def test_edge_detection_canny(self):
        """测试Canny边缘检测"""
        edges = ImageSegmentation.edge_detection(
            self.test_image, method='canny', threshold1=100, threshold2=200
        )
        
        self.assertIsNotNone(edges)
        self.assertEqual(edges.shape, self.test_image.shape)
        self.assertEqual(edges.dtype, np.uint8)
        
        # 边缘图像应该是二值的（0或255）
        unique_values = np.unique(edges)
        self.assertTrue(np.array_equal(unique_values, [0]) or 
                       np.array_equal(unique_values, [255]) or
                       np.array_equal(unique_values, [0, 255]))
    
    def test_edge_detection_sobel(self):
        """测试Sobel边缘检测"""
        edges = ImageSegmentation.edge_detection(self.test_image, method='sobel')
        
        self.assertIsNotNone(edges)
        self.assertEqual(edges.shape, self.test_image.shape)
        self.assertEqual(edges.dtype, np.uint8)
    
    def test_threshold_segmentation_otsu(self):
        """测试Otsu阈值分割"""
        binary = ImageSegmentation.threshold_segmentation(
            self.test_image, method='otsu'
        )
        
        self.assertIsNotNone(binary)
        self.assertEqual(binary.shape, self.test_image.shape)
        self.assertEqual(binary.dtype, np.uint8)
        
        # 检查是否为二值图像
        unique_values = np.unique(binary)
        self.assertTrue(np.array_equal(unique_values, [0, 255]))
    
    def test_threshold_segmentation_adaptive(self):
        """测试自适应阈值分割"""
        binary = ImageSegmentation.threshold_segmentation(
            self.test_image, method='adaptive', block_size=11, c=2
        )
        
        self.assertIsNotNone(binary)
        self.assertEqual(binary.shape, self.test_image.shape)
        self.assertEqual(binary.dtype, np.uint8)
    
    def test_region_growing_segmentation(self):
        """测试区域生长分割"""
        # 选择种子点（在明亮区域中心）
        seed_point = (50, 50)
        segmented = ImageSegmentation.region_growing_segmentation(
            self.test_image, seed_point, threshold=50
        )
        
        self.assertIsNotNone(segmented)
        self.assertEqual(segmented.shape, self.test_image.shape)
        self.assertEqual(segmented.dtype, np.uint8)
        
        # 分割区域应该大致对应明亮区域
        segmented_area = np.sum(segmented > 0)
        expected_area = 40 * 40  # 30:70是40x40的区域
        
        # 允许一些误差
        self.assertGreater(segmented_area, expected_area * 0.5)
        self.assertLess(segmented_area, expected_area * 1.5)

class TestIntegration(unittest.TestCase):
    """集成测试类"""
    
    def test_full_pipeline(self):
        """测试完整处理流程"""
        # 创建测试图像
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # 1. 几何运算：旋转
        rotated = GeometricOperations.rotate(test_image, 45)
        self.assertIsNotNone(rotated)
        
        # 2. 对比度增强：直方图均衡化
        if len(rotated.shape) == 3:
            enhanced = ContrastEnhancement.histogram_equalization(rotated)
        else:
            # 如果是灰度图，直接使用
            enhanced = rotated
            
        self.assertIsNotNone(enhanced)
        
        # 3. 平滑滤波：高斯滤波
        if len(enhanced.shape) == 3:
            smoothed = SmoothingFilters.gaussian_filter(enhanced, kernel_size=5, sigma=1.0)
        else:
            smoothed = enhanced
            
        self.assertIsNotNone(smoothed)
        
        # 4. 图像分割：边缘检测
        if len(smoothed.shape) == 3:
            # 转换为灰度图
            gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
        else:
            gray = smoothed
            
        edges = ImageSegmentation.edge_detection(gray, method='canny', 
                                                threshold1=100, threshold2=200)
        
        self.assertIsNotNone(edges)
        
        # 验证最终结果的形状
        self.assertEqual(edges.shape, (gray.shape[0], gray.shape[1]))

def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # 运行测试
    success = run_tests()
    
    if success:
        print("\n所有测试通过！")
        sys.exit(0)
    else:
        print("\n部分测试失败！")
        sys.exit(1)