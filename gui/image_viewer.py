#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像显示组件模块
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ImageViewer(ttk.Frame):
    """图像显示组件类"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.image = None
        self.image_tk = None
        self.canvas_image_id = None
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 创建Canvas用于显示图像
        self.canvas = tk.Canvas(self, bg='gray', highlightthickness=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        scrollbar_y = ttk.Scrollbar(self, orient=tk.VERTICAL)
        scrollbar_x = ttk.Scrollbar(self, orient=tk.HORIZONTAL)
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 配置Canvas滚动
        self.canvas.configure(yscrollcommand=scrollbar_y.set,
                             xscrollcommand=scrollbar_x.set)
        scrollbar_y.configure(command=self.canvas.yview)
        scrollbar_x.configure(command=self.canvas.xview)
        
        # 绑定鼠标事件
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        
        # 初始化变量
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start = None
        
    def display_image(self, image, is_rgb=True):
        """显示图像
    
        参数:
            image: 输入图像
            is_rgb: 如果为True，表示图像已经是RGB格式；如果为False，表示是BGR格式
        """
        self.image = image
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.is_rgb = is_rgb  # 保存色彩格式信息
        self.update_display()

    def update_display(self):
        """更新显示"""
        if self.image is None:
            return
        
        # 计算显示尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
    
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # 获取图像尺寸
        if len(self.image.shape) == 3:
            height, width, _ = self.image.shape
        else:
            height, width = self.image.shape
        
        # 计算缩放比例以适应Canvas
        scale_x = canvas_width / width
        scale_y = canvas_height / height
        self.scale = min(scale_x, scale_y, 1.0)  # 最大缩放为1
    
        # 缩放图像
        display_width = int(width * self.scale)
        display_height = int(height * self.scale)
    
        # 调整图像大小
        resized = cv2.resize(self.image, (display_width, display_height))
    
        # 转换为PIL图像 - 修复色彩转换
        if len(resized.shape) == 3:
            # 根据色彩格式决定如何转换
            if hasattr(self, 'is_rgb') and self.is_rgb:
                # 图像已经是RGB格式，直接使用
                rgb_image = resized
            else:
                # 图像是BGR格式，转换为RGB
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            # 灰度图像转换为RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        
        pil_image = Image.fromarray(rgb_image)
        self.image_tk = ImageTk.PhotoImage(pil_image)
    
        # 清除Canvas
        self.canvas.delete("all")
    
        # 显示图像
        if display_width < canvas_width:
            self.offset_x = (canvas_width - display_width) // 2
        else:
            self.offset_x = 0
        
        if display_height < canvas_height:
            self.offset_y = (canvas_height - display_height) // 2
        else:
            self.offset_y = 0
        
        self.canvas_image_id = self.canvas.create_image(
            self.offset_x, self.offset_y,
            anchor=tk.NW, image=self.image_tk
        )
    
        # 更新滚动区域
        self.canvas.config(scrollregion=(
            0, 0, max(canvas_width, display_width), 
            max(canvas_height, display_height)
        ))
        
    def draw_point(self, x, y, radius=5, color="red"):
        """在图像上绘制点"""
        if self.canvas_image_id is not None:
            # 计算实际坐标
            actual_x = x * self.scale + self.offset_x
            actual_y = y * self.scale + self.offset_y
            
            # 绘制圆形标记
            self.canvas.create_oval(
                actual_x - radius, actual_y - radius,
                actual_x + radius, actual_y + radius,
                fill=color, outline="yellow", width=2
            )
            
    def on_canvas_resize(self, event):
        """Canvas大小改变事件"""
        if self.image is not None:
            self.update_display()
            
    def on_mouse_press(self, event):
        """鼠标按下事件"""
        self.drag_start = (event.x, event.y)
        
    def on_mouse_drag(self, event):
        """鼠标拖动事件"""
        if self.drag_start and self.image is not None:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            
            # 移动Canvas视图
            self.canvas.xview_scroll(-dx, "units")
            self.canvas.yview_scroll(-dy, "units")
            
            self.drag_start = (event.x, event.y)
            
    def on_mouse_wheel(self, event):
        """鼠标滚轮事件"""
        if self.image is not None:
            # 缩放图像
            scale_factor = 1.1
            if event.delta > 0:
                self.scale *= scale_factor
            else:
                self.scale /= scale_factor
                
            # 限制缩放范围
            self.scale = max(0.1, min(5.0, self.scale))
            self.update_display()
            
    def bind(self, sequence, func):
        """绑定事件"""
        self.canvas.bind(sequence, func)
        
    def unbind(self, sequence):
        """解绑事件"""
        self.canvas.unbind(sequence)