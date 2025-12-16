#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主窗口模块
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

from gui.image_viewer import ImageViewer
from gui.controls import ControlPanel
from utils.image_loader import ImageLoader
from utils.histogram_utils import HistogramUtils
from algorithms.geometric_operations import GeometricOperations
from algorithms.contrast_enhancement import ContrastEnhancement
from algorithms.smoothing_filters import SmoothingFilters
from algorithms.image_segmentation import ImageSegmentation

class MainWindow:
    """主窗口类"""
    
    def __init__(self, root):
        self.root = root
        self.original_image = None
        self.processed_image = None
        self.history = []  # 历史记录
        self.history_index = -1
        
        self.setup_ui()
        self.bind_events()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        self.control_panel = ControlPanel(main_container, self)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 右侧图像显示区域
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建图像显示区域
        self.setup_image_display(right_frame)
        
        # 创建状态栏
        self.setup_status_bar()
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开图像", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="打开目录", command=self.open_directory)
        file_menu.add_separator()
        file_menu.add_command(label="保存图像", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="另存为", command=self.save_image_as)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit, accelerator="Ctrl+Q")
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 编辑菜单
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="撤销", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="重做", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="重置", command=self.reset_image, accelerator="Ctrl+R")
        menubar.add_cascade(label="编辑", menu=edit_menu)
        
        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="显示直方图", command=self.show_histogram)
        view_menu.add_command(label="显示图像信息", command=self.show_image_info)
        menubar.add_cascade(label="视图", menu=view_menu)
        
        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        
        # 几何运算子菜单
        geometric_menu = tk.Menu(tools_menu, tearoff=0)
        geometric_menu.add_command(label="平移", command=lambda: self.show_geometric_dialog('translate'))
        geometric_menu.add_command(label="旋转", command=lambda: self.show_geometric_dialog('rotate'))
        geometric_menu.add_command(label="缩放", command=lambda: self.show_geometric_dialog('scale'))
        geometric_menu.add_command(label="镜像", command=lambda: self.show_geometric_dialog('mirror'))
        tools_menu.add_cascade(label="几何运算", menu=geometric_menu)
        
        # 对比度增强子菜单
        contrast_menu = tk.Menu(tools_menu, tearoff=0)
        contrast_menu.add_command(label="线性变换", command=lambda: self.show_contrast_dialog('linear'))
        contrast_menu.add_command(label="Gamma校正", command=lambda: self.show_contrast_dialog('gamma'))
        contrast_menu.add_command(label="直方图均衡化", command=self.apply_histogram_equalization)
        contrast_menu.add_command(label="自适应直方图均衡化", command=self.apply_adaptive_histogram_equalization)
        tools_menu.add_cascade(label="对比度增强", menu=contrast_menu)
        
        # 平滑滤波子菜单
        smoothing_menu = tk.Menu(tools_menu, tearoff=0)
        smoothing_menu.add_command(label="均值滤波", command=lambda: self.show_filter_dialog('mean'))
        smoothing_menu.add_command(label="中值滤波", command=lambda: self.show_filter_dialog('median'))
        smoothing_menu.add_command(label="高斯滤波", command=lambda: self.show_filter_dialog('gaussian'))
        tools_menu.add_cascade(label="平滑滤波", menu=smoothing_menu)
        
        # 图像分割子菜单
        segmentation_menu = tk.Menu(tools_menu, tearoff=0)
        segmentation_menu.add_command(label="边缘检测", command=lambda: self.show_segmentation_dialog('edge'))
        segmentation_menu.add_command(label="阈值分割", command=lambda: self.show_segmentation_dialog('threshold'))
        segmentation_menu.add_command(label="区域生长", command=lambda: self.show_segmentation_dialog('region'))
        tools_menu.add_cascade(label="图像分割", menu=segmentation_menu)
        
        menubar.add_cascade(label="工具", menu=tools_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        # 绑定快捷键
        self.root.bind('<Control-o>', lambda e: self.open_image())
        self.root.bind('<Control-s>', lambda e: self.save_image())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Control-r>', lambda e: self.reset_image())
        
    def setup_image_display(self, parent):
        """设置图像显示区域"""
        # 创建显示框架
        display_frame = ttk.Frame(parent)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # 原图像显示
        orig_frame = ttk.LabelFrame(display_frame, text="原图像")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.orig_viewer = ImageViewer(orig_frame)
        self.orig_viewer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 处理后图像显示
        proc_frame = ttk.LabelFrame(display_frame, text="处理后图像")
        proc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.proc_viewer = ImageViewer(proc_frame)
        self.proc_viewer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        
    def setup_status_bar(self):
        """设置状态栏"""
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, height=25)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="就绪")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.image_info_label = ttk.Label(self.status_bar, text="")
        self.image_info_label.pack(side=tk.RIGHT, padx=5)
        
    def bind_events(self):
        """绑定事件"""
        # 图像查看器事件
        self.orig_viewer.bind("<Button-1>", self.on_image_click)
        
    # ========== 文件操作相关方法 ==========
    
    def open_image(self):
        """打开图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("PNG文件", "*.png"),
                ("BMP文件", "*.bmp"),
                ("TIFF文件", "*.tiff *.tif"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.original_image = ImageLoader.load_image(file_path)
                if self.original_image is not None:
                    # 修复：确保图像是RGB格式
                    if len(self.original_image.shape) == 3:
                        # 检查是否为BGR（OpenCV默认）
                        # 简单检查：如果蓝色通道值大于红色通道
                        if self.original_image[0, 0, 0] > self.original_image[0, 0, 2]:
                            # 是BGR，转换为RGB
                            import cv2
                            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                    self.processed_image = self.original_image.copy()
                    self.update_image_display()
                    self.update_status(f"已加载图像: {os.path.basename(file_path)}")
                    self.update_image_info()
                    # 清空历史记录
                    self.history = [self.original_image.copy()]
                    self.history_index = 0
            except Exception as e:
                messagebox.showerror("错误", f"加载图像失败: {str(e)}")
                
    def open_directory(self):
        """打开图像目录"""
        dir_path = filedialog.askdirectory(title="选择图像目录")
        if dir_path:
            # TODO: 实现目录浏览功能
            messagebox.showinfo("提示", "目录浏览功能待实现")
            
    def save_image(self):
        """保存图像"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "没有图像可以保存")
            return
            
        if hasattr(self, 'last_save_path'):
            self._save_image_to_path(self.last_save_path)
        else:
            self.save_image_as()
            
    def save_image_as(self):
        """另存为图像"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "没有图像可以保存")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="保存图像",
            defaultextension=".png",
            filetypes=[
                ("PNG文件", "*.png"),
                ("JPEG文件", "*.jpg"),
                ("BMP文件", "*.bmp"),
                ("TIFF文件", "*.tiff"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self._save_image_to_path(file_path)
            
    def _save_image_to_path(self, file_path):
        """保存图像到指定路径"""
        try:
            # 复制图像
            save_image = self.processed_image.copy()
        
            # 如果是RGB，转换为BGR保存
            if len(save_image.shape) == 3:
                # 检查是否为RGB（红色通道大于蓝色通道）
                if save_image[0, 0, 0] < save_image[0, 0, 2]:
                    # 是RGB，转换为BGR
                    import cv2
                    save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        
            cv2.imwrite(file_path, save_image)
            self.last_save_path = file_path
            self.update_status(f"图像已保存: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"保存图像失败: {str(e)}")
            
    # ========== 编辑操作相关方法 ==========
    
    def undo(self):
        """撤销操作"""
        if self.history_index > 0:
            self.history_index -= 1
            self.processed_image = self.history[self.history_index].copy()
            self.update_image_display()
            self.update_status("已撤销上一步操作")
            
    def redo(self):
        """重做操作"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.processed_image = self.history[self.history_index].copy()
            self.update_image_display()
            self.update_status("已重做操作")
            
    def reset_image(self):
        """重置图像"""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_image_display()
            self.update_status("图像已重置")
            # 清空历史记录
            self.history = [self.original_image.copy()]
            self.history_index = 0
            
    def add_to_history(self):
        """添加到历史记录"""
        if self.processed_image is not None:
            # 移除当前索引之后的历史记录
            self.history = self.history[:self.history_index + 1]
            # 添加新记录
            self.history.append(self.processed_image.copy())
            self.history_index += 1
            
    # ========== 图像处理相关方法 ==========
    
    def apply_geometric_operation(self, operation, **kwargs):
        """应用几何运算"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
            
        try:
            result = getattr(GeometricOperations, operation)(self.processed_image, **kwargs)
            if result is not None:
                self.processed_image = result
                self.update_image_display()
                self.add_to_history()
                self.update_status(f"已应用{operation}操作")
        except Exception as e:
            messagebox.showerror("错误", f"几何运算失败: {str(e)}")
            
    def apply_contrast_enhancement(self, method, **kwargs):
        """应用对比度增强"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
            
        try:
            if method == 'histogram_equalization':
                result = ContrastEnhancement.histogram_equalization(self.processed_image)
            elif method == 'adaptive_histogram_equalization':
                result = ContrastEnhancement.adaptive_histogram_equalization(self.processed_image, **kwargs)
            else:
                result = ContrastEnhancement.grayscale_transform(self.processed_image, method, **kwargs)
                
            if result is not None:
                self.processed_image = result
                self.update_image_display()
                self.add_to_history()
                self.update_status(f"已应用{method}增强")
        except Exception as e:
            messagebox.showerror("错误", f"对比度增强失败: {str(e)}")
            
    def apply_filter(self, filter_type, **kwargs):
        """应用滤波器"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
            
        try:
            result = getattr(SmoothingFilters, f"{filter_type}_filter")(self.processed_image, **kwargs)
            if result is not None:
                self.processed_image = result
                self.update_image_display()
                self.add_to_history()
                self.update_status(f"已应用{filter_type}滤波")
        except Exception as e:
            messagebox.showerror("错误", f"滤波操作失败: {str(e)}")
            
    def apply_segmentation(self, segmentation_type, **kwargs):
        """应用图像分割 - 修复版"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        try:
            # 根据segmentation_type决定调用哪个方法
            if segmentation_type == 'edge_detection':
                # 获取edge_detection的method参数
                edge_method = kwargs.get('method', 'canny')
                # 移除method参数，避免重复传递
                if 'method' in kwargs:
                    del kwargs['method']
                result = ImageSegmentation.edge_detection(
                    self.processed_image, 
                    method=edge_method, 
                    **kwargs
                )
            
            elif segmentation_type == 'threshold_segmentation':
                # 获取threshold_segmentation的method参数
                threshold_method = kwargs.get('method', 'otsu')
                if 'method' in kwargs:
                    del kwargs['method']
                result = ImageSegmentation.threshold_segmentation(
                    self.processed_image,
                    method=threshold_method,
                    **kwargs
                )
            
            elif segmentation_type == 'region_growing_segmentation':
                result = ImageSegmentation.region_growing_segmentation(
                    self.processed_image,
                    **kwargs
                )
            else:
                raise ValueError(f"不支持的分割类型: {segmentation_type}")
            
            if result is not None:
                self.processed_image = result
                self.update_image_display()
                self.add_to_history()
                self.update_status(f"已应用{segmentation_type}")
        except Exception as e:
            messagebox.showerror("错误", f"图像分割失败: {str(e)}")
            
    # ========== 对话框相关方法 ==========
    
    def show_geometric_dialog(self, operation):
        """显示几何运算对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{operation}设置")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 根据操作类型创建不同的控件
        if operation == 'translate':
            ttk.Label(dialog, text="X方向位移:").pack(pady=5)
            dx_var = tk.IntVar(value=0)
            ttk.Scale(dialog, from_=-100, to=100, variable=dx_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
            
            ttk.Label(dialog, text="Y方向位移:").pack(pady=5)
            dy_var = tk.IntVar(value=0)
            ttk.Scale(dialog, from_=-100, to=100, variable=dy_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
            
            def apply_translation():
                self.apply_geometric_operation('translate', dx=dx_var.get(), dy=dy_var.get())
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_translation).pack(pady=10)
            
        elif operation == 'rotate':
            ttk.Label(dialog, text="旋转角度:").pack(pady=5)
            angle_var = tk.DoubleVar(value=0)
            ttk.Scale(dialog, from_=-180, to=180, variable=angle_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
            
            def apply_rotation():
                self.apply_geometric_operation('rotate', angle=angle_var.get())
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_rotation).pack(pady=10)
            
        elif operation == 'scale':
            ttk.Label(dialog, text="缩放比例:").pack(pady=5)
            scale_var = tk.DoubleVar(value=1.0)
            ttk.Scale(dialog, from_=0.1, to=3.0, variable=scale_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
            
            def apply_scaling():
                self.apply_geometric_operation('scale', fx=scale_var.get(), fy=scale_var.get())
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_scaling).pack(pady=10)
            
        elif operation == 'mirror':
            ttk.Label(dialog, text="选择镜像方式:").pack(pady=10)
            
            var = tk.StringVar(value='horizontal')
            ttk.Radiobutton(dialog, text="水平镜像", variable=var, value='horizontal').pack()
            ttk.Radiobutton(dialog, text="垂直镜像", variable=var, value='vertical').pack()
            ttk.Radiobutton(dialog, text="对角镜像", variable=var, value='both').pack()
            
            def apply_mirror():
                self.apply_geometric_operation('mirror', mode=var.get())
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_mirror).pack(pady=10)
            
    def show_contrast_dialog(self, method):
        """显示对比度增强对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{method}设置")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        if method == 'gamma':
            ttk.Label(dialog, text="Gamma值:").pack(pady=5)
            gamma_var = tk.DoubleVar(value=1.0)
            ttk.Scale(dialog, from_=0.1, to=3.0, variable=gamma_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
            
            def apply_gamma():
                self.apply_contrast_enhancement('gamma', gamma=gamma_var.get())
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_gamma).pack(pady=10)
            
        elif method == 'linear':
            ttk.Label(dialog, text="Alpha值:").pack(pady=5)
            alpha_var = tk.DoubleVar(value=1.0)
            ttk.Scale(dialog, from_=0.1, to=3.0, variable=alpha_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
            
            ttk.Label(dialog, text="Beta值:").pack(pady=5)
            beta_var = tk.IntVar(value=0)
            ttk.Scale(dialog, from_=-100, to=100, variable=beta_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
            
            def apply_linear():
                # 临时使用，实际应调用对应方法
                import cv2
                result = cv2.convertScaleAbs(self.processed_image, 
                                            alpha=alpha_var.get(), 
                                            beta=beta_var.get())
                self.processed_image = result
                self.update_image_display()
                self.add_to_history()
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_linear).pack(pady=10)
            
    def show_filter_dialog(self, filter_type):
        """显示滤波对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{filter_type}滤波设置")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="滤波器大小:").pack(pady=5)
        size_var = tk.IntVar(value=3)
        ttk.Scale(dialog, from_=3, to=15, variable=size_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
        
        if filter_type == 'gaussian':
            ttk.Label(dialog, text="Sigma值:").pack(pady=5)
            sigma_var = tk.DoubleVar(value=0)
            ttk.Scale(dialog, from_=0, to=10, variable=sigma_var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)
            
            def apply_gaussian():
                self.apply_filter('gaussian', kernel_size=size_var.get(), sigma=sigma_var.get())
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_gaussian).pack(pady=10)
        else:
            def apply_filter():
                self.apply_filter(filter_type, kernel_size=size_var.get())
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_filter).pack(pady=10)
            
    def show_segmentation_dialog(self, method):
        """显示分割对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"{method}分割设置")
        dialog.geometry("300x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        if method == 'edge_detection':
            ttk.Label(dialog, text="选择边缘检测方法:").pack(pady=10)
            
            var = tk.StringVar(value='canny')
            ttk.Radiobutton(dialog, text="Canny边缘检测", variable=var, value='canny').pack()
            ttk.Radiobutton(dialog, text="Sobel边缘检测", variable=var, value='sobel').pack()
            ttk.Radiobutton(dialog, text="Laplacian边缘检测", variable=var, value='laplacian').pack()
            
            def apply_edge_detection():
                if var.get() == 'canny':
                    self.apply_segmentation('edge_detection', method='canny', 
                                           threshold1=100, threshold2=200)
                else:
                    self.apply_segmentation('edge_detection', method=var.get())
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_edge_detection).pack(pady=10)
            
        elif method == 'threshold_segmentation':
            ttk.Label(dialog, text="选择阈值方法:").pack(pady=10)
            
            var = tk.StringVar(value='otsu')
            ttk.Radiobutton(dialog, text="Otsu阈值", variable=var, value='otsu').pack()
            ttk.Radiobutton(dialog, text="自适应阈值", variable=var, value='adaptive').pack()
            ttk.Radiobutton(dialog, text="全局阈值", variable=var, value='global').pack()
            
            def apply_threshold():
                self.apply_segmentation('threshold_segmentation', method=var.get())
                dialog.destroy()
                
            ttk.Button(dialog, text="应用", command=apply_threshold).pack(pady=10)
            
        elif method == 'region_growing_segmentation':
            ttk.Label(dialog, text="请点击图像选择种子点").pack(pady=10)
            
            self.seed_point = None
            self.segmentation_dialog = dialog
            
            # 绑定点击事件
            self.orig_viewer.bind("<Button-1>", self.on_seed_select)
            
    # ========== 视图相关方法 ==========
    
    def show_histogram(self):
        """显示直方图"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
            
        dialog = tk.Toplevel(self.root)
        dialog.title("图像直方图")
        dialog.geometry("800x600")
        
        # 创建直方图
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # 原图像直方图
        if self.original_image is not None:
            HistogramUtils.plot_histogram(self.original_image, axes[0])
            axes[0].set_title("原图像直方图")
        
        # 处理后图像直方图
        HistogramUtils.plot_histogram(self.processed_image, axes[1])
        axes[1].set_title("处理后图像直方图")
        
        plt.tight_layout()
        
        # 在Tkinter中显示
        canvas = FigureCanvasTkAgg(fig, master=dialog)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=10)
        
    def show_image_info(self):
        """显示图像信息"""
        if self.processed_image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return
            
        info = f"图像尺寸: {self.processed_image.shape[1]} x {self.processed_image.shape[0]}\n"
        info += f"通道数: {self.processed_image.shape[2] if len(self.processed_image.shape) == 3 else 1}\n"
        info += f"数据类型: {self.processed_image.dtype}\n"
        
        if len(self.processed_image.shape) == 2:
            info += f"像素范围: [{self.processed_image.min()}, {self.processed_image.max()}]"
        else:
            info += "像素范围: 多通道"
            
        messagebox.showinfo("图像信息", info)
        
    def toggle_compare_mode(self):
        """切换对比显示模式"""
        # TODO: 实现对比显示模式
        messagebox.showinfo("提示", "对比显示模式待实现")
        
    # ========== 帮助相关方法 ==========
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """数字图像处理平台使用说明

1. 文件操作
   - 打开图像: 从文件系统加载图像
   - 保存图像: 保存处理后的图像
   - 打开目录: 浏览图像目录

2. 编辑操作
   - 撤销/重做: 撤销或重做图像处理操作
   - 重置: 恢复原始图像

3. 图像处理工具
   - 几何运算: 平移、旋转、缩放、镜像
   - 对比度增强: 灰度变换、直方图均衡化
   - 平滑滤波: 均值、中值、高斯滤波
   - 图像分割: 边缘检测、阈值分割、区域生长

4. 视图选项
   - 显示直方图: 查看图像直方图
   - 图像信息: 查看图像详细信息

快捷键:
  Ctrl+O: 打开图像
  Ctrl+S: 保存图像
  Ctrl+Z: 撤销
  Ctrl+Y: 重做
  Ctrl+R: 重置
  Ctrl+Q: 退出"""
        
        messagebox.showinfo("使用说明", help_text)
        
    def show_about(self):
        """显示关于信息"""
        about_text = """数字图像处理平台 v1.0.0

课程实践项目 - 数字图像处理算法系统的设计及实现

功能特点:
- 完整的图像处理算法实现
- 直观的图形用户界面
- 实时预览处理效果
- 支持多种图像格式
- 操作历史记录

作者: 李子豪
日期: 2025年
课程: 数字图像处理"""
        
        messagebox.showinfo("关于", about_text)
        
    # ========== 事件处理相关方法 ==========
    
    def on_image_click(self, event):
        """图像点击事件处理"""
        if hasattr(self, 'seed_point_waiting'):
            self.seed_point = (event.y, event.x)  # 注意坐标顺序
            self.orig_viewer.unbind("<Button-1>")
            
            # 显示种子点
            self.orig_viewer.draw_point(event.x, event.y)
            
            # 应用区域生长分割
            if self.seed_point:
                threshold = 10  # 默认阈值
                self.apply_segmentation('region_growing_segmentation', 
                                       seed_point=self.seed_point, 
                                       threshold=threshold)
            
            if hasattr(self, 'segmentation_dialog'):
                self.segmentation_dialog.destroy()
                
    def on_seed_select(self, event):
        """选择种子点事件"""
        self.on_image_click(event)
        
    # ========== 工具方法 ==========
    
    def update_image_display(self):
        """更新图像显示"""
        if self.original_image is not None:
            self.orig_viewer.display_image(self.original_image)
            
        if self.processed_image is not None:
            self.proc_viewer.display_image(self.processed_image)
            
    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
        
    def update_image_info(self):
        """更新图像信息"""
        if self.original_image is not None:
            info = f"{self.original_image.shape[1]}x{self.original_image.shape[0]}"
            if len(self.original_image.shape) == 3:
                info += f" ({self.original_image.shape[2]}通道)"
            else:
                info += " (灰度)"
            self.image_info_label.config(text=info)
            
    def apply_histogram_equalization(self):
        """应用直方图均衡化"""
        self.apply_contrast_enhancement('histogram_equalization')
        
    def apply_adaptive_histogram_equalization(self):
        """应用自适应直方图均衡化"""
        self.apply_contrast_enhancement('adaptive_histogram_equalization')