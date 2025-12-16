#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
控制面板模块
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np  # 添加这行

class ControlPanel(ttk.Frame):
    """控制面板类"""
    
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 几何运算控制组
        self.create_geometric_controls()
        
        # 对比度增强控制组
        self.create_contrast_controls()
        
        # 平滑滤波控制组
        self.create_filter_controls()
        
        # 图像分割控制组
        self.create_segmentation_controls()
        
    def create_geometric_controls(self):
        """创建几何运算控制组"""
        frame = ttk.LabelFrame(self, text="几何运算", padding="10")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 平移控制
        translate_frame = ttk.Frame(frame)
        translate_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(translate_frame, text="平移:").pack(side=tk.LEFT)
        self.dx_var = tk.IntVar(value=0)
        self.dy_var = tk.IntVar(value=0)
        
        ttk.Entry(translate_frame, textvariable=self.dx_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(translate_frame, text="X").pack(side=tk.LEFT)
        ttk.Entry(translate_frame, textvariable=self.dy_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(translate_frame, text="Y").pack(side=tk.LEFT)
        ttk.Button(translate_frame, text="应用", 
                  command=self.apply_translation).pack(side=tk.LEFT, padx=5)
        
        # 旋转控制
        rotate_frame = ttk.Frame(frame)
        rotate_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(rotate_frame, text="旋转角度:").pack(side=tk.LEFT)
        self.angle_var = tk.DoubleVar(value=0)
        ttk.Scale(rotate_frame, from_=-180, to=180, variable=self.angle_var, 
                 orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        ttk.Button(rotate_frame, text="应用", 
                  command=self.apply_rotation).pack(side=tk.LEFT)
        
        # 缩放控制
        scale_frame = ttk.Frame(frame)
        scale_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(scale_frame, text="缩放比例:").pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=1.0)
        ttk.Scale(scale_frame, from_=0.1, to=3.0, variable=self.scale_var,
                 orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        ttk.Button(scale_frame, text="应用",
                  command=self.apply_scaling).pack(side=tk.LEFT)
        
        # 镜像控制
        mirror_frame = ttk.Frame(frame)
        mirror_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mirror_frame, text="镜像:").pack(side=tk.LEFT)
        self.mirror_var = tk.StringVar(value='horizontal')
        ttk.Radiobutton(mirror_frame, text="水平", variable=self.mirror_var,
                       value='horizontal').pack(side=tk.LEFT)
        ttk.Radiobutton(mirror_frame, text="垂直", variable=self.mirror_var,
                       value='vertical').pack(side=tk.LEFT)
        ttk.Radiobutton(mirror_frame, text="对角", variable=self.mirror_var,
                       value='both').pack(side=tk.LEFT)
        ttk.Button(mirror_frame, text="应用",
                  command=self.apply_mirror).pack(side=tk.LEFT, padx=5)
        
    def create_contrast_controls(self):
        """创建对比度增强控制组"""
        frame = ttk.LabelFrame(self, text="对比度增强", padding="10")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 灰度变换
        grayscale_frame = ttk.Frame(frame)
        grayscale_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(grayscale_frame, text="灰度变换:").pack(side=tk.LEFT)
        self.gamma_var = tk.DoubleVar(value=1.0)
        ttk.Scale(grayscale_frame, from_=0.1, to=3.0, variable=self.gamma_var,
                 orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=5)
        ttk.Button(grayscale_frame, text="Gamma校正",
                  command=self.apply_gamma).pack(side=tk.LEFT)
        
        # 直方图均衡化
        hist_frame = ttk.Frame(frame)
        hist_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(hist_frame, text="直方图均衡化",
                  command=self.main_window.apply_histogram_equalization).pack(side=tk.LEFT)
        ttk.Button(hist_frame, text="自适应直方图均衡化",
                  command=self.main_window.apply_adaptive_histogram_equalization).pack(side=tk.LEFT, padx=5)
        
    def create_filter_controls(self):
        """创建平滑滤波控制组"""
        frame = ttk.LabelFrame(self, text="平滑滤波", padding="10")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 滤波器类型选择
        filter_frame = ttk.Frame(frame)
        filter_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(filter_frame, text="滤波器:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar(value='mean')
        ttk.Radiobutton(filter_frame, text="均值", variable=self.filter_var,
                       value='mean').pack(side=tk.LEFT)
        ttk.Radiobutton(filter_frame, text="中值", variable=self.filter_var,
                       value='median').pack(side=tk.LEFT)
        ttk.Radiobutton(filter_frame, text="高斯", variable=self.filter_var,
                       value='gaussian').pack(side=tk.LEFT)
        
        # 滤波器参数
        param_frame = ttk.Frame(frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="大小:").pack(side=tk.LEFT)
        self.filter_size_var = tk.IntVar(value=3)
        ttk.Spinbox(param_frame, from_=3, to=15, textvariable=self.filter_size_var,
                   width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(param_frame, text="Sigma:").pack(side=tk.LEFT)
        self.sigma_var = tk.DoubleVar(value=0)
        ttk.Spinbox(param_frame, from_=0, to=10, textvariable=self.sigma_var,
                   width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(param_frame, text="应用滤波",
                  command=self.apply_filter).pack(side=tk.LEFT, padx=5)
        
    def create_segmentation_controls(self):
        """创建图像分割控制组"""
        frame = ttk.LabelFrame(self, text="图像分割", padding="10")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 边缘检测
        edge_frame = ttk.Frame(frame)
        edge_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(edge_frame, text="边缘检测:").pack(side=tk.LEFT)
        self.edge_var = tk.StringVar(value='canny')
        ttk.Radiobutton(edge_frame, text="Canny", variable=self.edge_var,
                       value='canny').pack(side=tk.LEFT)
        ttk.Radiobutton(edge_frame, text="Sobel", variable=self.edge_var,
                       value='sobel').pack(side=tk.LEFT)
        ttk.Radiobutton(edge_frame, text="Laplacian", variable=self.edge_var,
                       value='laplacian').pack(side=tk.LEFT)
        ttk.Button(edge_frame, text="应用",
                  command=self.apply_edge_detection).pack(side=tk.LEFT, padx=5)
        
        # 阈值分割
        threshold_frame = ttk.Frame(frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(threshold_frame, text="阈值分割:").pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value='otsu')
        ttk.Radiobutton(threshold_frame, text="Otsu", variable=self.threshold_var,
                       value='otsu').pack(side=tk.LEFT)
        ttk.Radiobutton(threshold_frame, text="自适应", variable=self.threshold_var,
                       value='adaptive').pack(side=tk.LEFT)
        ttk.Radiobutton(threshold_frame, text="全局", variable=self.threshold_var,
                       value='global').pack(side=tk.LEFT)
        ttk.Button(threshold_frame, text="应用",
                  command=self.apply_threshold).pack(side=tk.LEFT, padx=5)
        
        # 阈值参数
        thresh_param_frame = ttk.Frame(frame)
        thresh_param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(thresh_param_frame, text="阈值:").pack(side=tk.LEFT)
        self.thresh_value_var = tk.IntVar(value=127)
        ttk.Scale(thresh_param_frame, from_=0, to=255, variable=self.thresh_value_var,
                 orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        
    # ========== 控制按钮回调方法 ==========
    
    def apply_translation(self):
        """应用平移"""
        self.main_window.apply_geometric_operation(
            'translate',
            dx=self.dx_var.get(),
            dy=self.dy_var.get()
        )
        
    def apply_rotation(self):
        """应用旋转"""
        self.main_window.apply_geometric_operation(
            'rotate',
            angle=self.angle_var.get()
        )
        
    def apply_scaling(self):
        """应用缩放"""
        self.main_window.apply_geometric_operation(
            'scale',
            fx=self.scale_var.get(),
            fy=self.scale_var.get()
        )
        
    def apply_mirror(self):
        """应用镜像"""
        self.main_window.apply_geometric_operation(
            'mirror',
            mode=self.mirror_var.get()
        )
        
    def apply_gamma(self):
        """应用Gamma校正"""
        self.main_window.apply_contrast_enhancement(
            'gamma',
            gamma=self.gamma_var.get()
        )
        
    def apply_filter(self):
        """应用滤波"""
        filter_type = self.filter_var.get()
        if filter_type == 'gaussian':
            self.main_window.apply_filter(
                filter_type,
                kernel_size=self.filter_size_var.get(),
                sigma=self.sigma_var.get()
            )
        else:
            self.main_window.apply_filter(
                filter_type,
                kernel_size=self.filter_size_var.get()
            )
            
    def apply_edge_detection(self):
        """应用边缘检测"""
        if self.edge_var.get() == 'canny':
            # 计算自动阈值
            if self.main_window.processed_image is not None:
                if len(self.main_window.processed_image.shape) == 3:
                    gray = cv2.cvtColor(self.main_window.processed_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = self.main_window.processed_image
                
                median = np.median(gray)
                lower = int(max(0, 0.7 * median))
                upper = int(min(255, 1.3 * median))
            
                # 调用修改后的apply_segmentation
                self.main_window.apply_segmentation(
                    'edge_detection',  # segmentation_type
                    method='canny',    # 传递给edge_detection的参数
                    threshold1=lower,
                    threshold2=upper
                )
        else:
            self.main_window.apply_segmentation(
                'edge_detection',      # segmentation_type
                method=self.edge_var.get()  # 传递给edge_detection的参数
            )
        
    def apply_threshold(self):
        """应用阈值分割"""
        if self.threshold_var.get() == 'global':
            self.main_window.apply_segmentation(
                'threshold_segmentation',  # segmentation_type
                method='global',           # 传递给threshold_segmentation的参数
                threshold=self.thresh_value_var.get()
            )
        else:
            self.main_window.apply_segmentation(
                'threshold_segmentation',  # segmentation_type
                method=self.threshold_var.get()  # 传递给threshold_segmentation的参数
            )