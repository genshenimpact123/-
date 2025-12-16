#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件操作工具模块
提供文件系统操作、路径管理等功能
"""

import os
import json
import pickle
import shutil
from datetime import datetime
import glob

class FileOperations:
    """文件操作工具类"""
    
    @staticmethod
    def ensure_directory(directory_path):
        """
        确保目录存在，如果不存在则创建
        
        参数:
            directory_path: 目录路径
        """
        os.makedirs(directory_path, exist_ok=True)
    
    @staticmethod
    def get_file_list(directory_path, pattern="*"):
        """
        获取目录中的文件列表
        
        参数:
            directory_path: 目录路径
            pattern: 文件匹配模式
            
        返回:
            文件路径列表
        """
        if not os.path.exists(directory_path):
            return []
        
        file_list = glob.glob(os.path.join(directory_path, pattern))
        # 过滤掉目录
        file_list = [f for f in file_list if os.path.isfile(f)]
        
        return sorted(file_list)
    
    @staticmethod
    def get_image_files(directory_path):
        """
        获取目录中的图像文件列表
        
        参数:
            directory_path: 目录路径
            
        返回:
            图像文件路径列表
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']
        all_files = FileOperations.get_file_list(directory_path)
        
        image_files = []
        for file_path in all_files:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in image_extensions:
                image_files.append(file_path)
        
        return image_files
    
    @staticmethod
    def create_backup(file_path):
        """
        创建文件备份
        
        参数:
            file_path: 文件路径
            
        返回:
            备份文件路径，如果备份失败则返回None
        """
        if not os.path.exists(file_path):
            return None
        
        # 生成备份文件名（添加时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(file_path)
        name_without_ext, ext = os.path.splitext(base_name)
        backup_name = f"{name_without_ext}_backup_{timestamp}{ext}"
        
        # 备份目录
        backup_dir = os.path.join(os.path.dirname(file_path), "backups")
        FileOperations.ensure_directory(backup_dir)
        
        backup_path = os.path.join(backup_dir, backup_name)
        
        try:
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"备份失败: {str(e)}")
            return None
    
    @staticmethod
    def save_json(data, file_path, indent=4):
        """
        保存数据为JSON文件
        
        参数:
            data: 要保存的数据
            file_path: 文件路径
            indent: 缩进空格数
        """
        FileOperations.ensure_directory(os.path.dirname(file_path))
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            return True
        except Exception as e:
            print(f"保存JSON失败: {str(e)}")
            return False
    
    @staticmethod
    def load_json(file_path):
        """
        从JSON文件加载数据
        
        参数:
            file_path: 文件路径
            
        返回:
            加载的数据，如果失败则返回None
        """
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"加载JSON失败: {str(e)}")
            return None
    
    @staticmethod
    def save_pickle(data, file_path):
        """
        保存数据为pickle文件
        
        参数:
            data: 要保存的数据
            file_path: 文件路径
        """
        FileOperations.ensure_directory(os.path.dirname(file_path))
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"保存pickle失败: {str(e)}")
            return False
    
    @staticmethod
    def load_pickle(file_path):
        """
        从pickle文件加载数据
        
        参数:
            file_path: 文件路径
            
        返回:
            加载的数据，如果失败则返回None
        """
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"加载pickle失败: {str(e)}")
            return None
    
    @staticmethod
    def get_file_info(file_path):
        """
        获取文件信息
        
        参数:
            file_path: 文件路径
            
        返回:
            文件信息字典
        """
        if not os.path.exists(file_path):
            return None
        
        try:
            stat_info = os.stat(file_path)
            
            info = {
                'path': file_path,
                'name': os.path.basename(file_path),
                'size': stat_info.st_size,
                'size_human': FileOperations._bytes_to_human(stat_info.st_size),
                'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                'accessed': datetime.fromtimestamp(stat_info.st_atime).isoformat(),
                'extension': os.path.splitext(file_path)[1].lower(),
                'directory': os.path.dirname(file_path)
            }
            
            return info
        except Exception as e:
            print(f"获取文件信息失败: {str(e)}")
            return None
    
    @staticmethod
    def _bytes_to_human(size_bytes):
        """将字节数转换为人类可读的格式"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.2f} {size_names[i]}"
    
    @staticmethod
    def copy_file(src_path, dst_path, overwrite=False):
        """
        复制文件
        
        参数:
            src_path: 源文件路径
            dst_path: 目标文件路径
            overwrite: 是否覆盖已存在的文件
            
        返回:
            是否成功
        """
        if not os.path.exists(src_path):
            return False
        
        if os.path.exists(dst_path) and not overwrite:
            return False
        
        try:
            FileOperations.ensure_directory(os.path.dirname(dst_path))
            shutil.copy2(src_path, dst_path)
            return True
        except Exception as e:
            print(f"复制文件失败: {str(e)}")
            return False
    
    @staticmethod
    def move_file(src_path, dst_path, overwrite=False):
        """
        移动文件
        
        参数:
            src_path: 源文件路径
            dst_path: 目标文件路径
            overwrite: 是否覆盖已存在的文件
            
        返回:
            是否成功
        """
        if not os.path.exists(src_path):
            return False
        
        if os.path.exists(dst_path) and not overwrite:
            return False
        
        try:
            FileOperations.ensure_directory(os.path.dirname(dst_path))
            shutil.move(src_path, dst_path)
            return True
        except Exception as e:
            print(f"移动文件失败: {str(e)}")
            return False
    
    @staticmethod
    def delete_file(file_path):
        """
        删除文件
        
        参数:
            file_path: 文件路径
            
        返回:
            是否成功
        """
        if not os.path.exists(file_path):
            return False
        
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            print(f"删除文件失败: {str(e)}")
            return False
    
    @staticmethod
    def rename_file(old_path, new_name):
        """
        重命名文件
        
        参数:
            old_path: 原文件路径
            new_name: 新文件名（不含路径）
            
        返回:
            新文件路径，如果失败则返回None
        """
        if not os.path.exists(old_path):
            return None
        
        directory = os.path.dirname(old_path)
        new_path = os.path.join(directory, new_name)
        
        try:
            os.rename(old_path, new_path)
            return new_path
        except Exception as e:
            print(f"重命名文件失败: {str(e)}")
            return None
    
    @staticmethod
    def get_unique_filename(directory, base_name, extension):
        """
        获取唯一的文件名（避免重复）
        
        参数:
            directory: 目录路径
            base_name: 基础文件名
            extension: 文件扩展名（含点）
            
        返回:
            唯一的文件路径
        """
        FileOperations.ensure_directory(directory)
        
        counter = 1
        while True:
            if counter == 1:
                filename = f"{base_name}{extension}"
            else:
                filename = f"{base_name}_{counter}{extension}"
            
            file_path = os.path.join(directory, filename)
            
            if not os.path.exists(file_path):
                return file_path
            
            counter += 1
    
    @staticmethod
    def create_project_structure(base_path):
        """
        创建项目目录结构
        
        参数:
            base_path: 基础路径
            
        返回:
            是否成功
        """
        try:
            directories = [
                "images",
                "output",
                "output/processed",
                "output/results",
                "output/thumbnails",
                "config",
                "logs",
                "backups",
                "tests",
                "docs"
            ]
            
            for directory in directories:
                dir_path = os.path.join(base_path, directory)
                FileOperations.ensure_directory(dir_path)
            
            # 创建README文件
            readme_path = os.path.join(base_path, "README.md")
            if not os.path.exists(readme_path):
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write("# 数字图像处理平台\n\n")
                    f.write("这是一个用于数字图像处理的综合平台。\n\n")
                    f.write("## 目录结构\n")
                    f.write("- images/: 存放输入图像\n")
                    f.write("- output/: 存放处理结果\n")
                    f.write("- config/: 配置文件\n")
                    f.write("- logs/: 日志文件\n")
                    f.write("- backups/: 备份文件\n")
            
            return True
        except Exception as e:
            print(f"创建项目结构失败: {str(e)}")
            return False
    
    @staticmethod
    def save_image_metadata(image_path, metadata, metadata_file=None):
        """
        保存图像元数据
        
        参数:
            image_path: 图像文件路径
            metadata: 元数据字典
            metadata_file: 元数据文件路径，如果为None则自动生成
            
        返回:
            是否成功
        """
        if metadata_file is None:
            # 自动生成元数据文件路径
            base_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(base_name)[0]
            metadata_file = os.path.join(
                os.path.dirname(image_path),
                f"{name_without_ext}_metadata.json"
            )
        
        # 添加时间戳
        metadata['save_time'] = datetime.now().isoformat()
        metadata['image_file'] = os.path.basename(image_path)
        
        return FileOperations.save_json(metadata, metadata_file)
    
    @staticmethod
    def load_image_metadata(image_path):
        """
        加载图像元数据
        
        参数:
            image_path: 图像文件路径
            
        返回:
            元数据字典，如果失败则返回None
        """
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        metadata_file = os.path.join(
            os.path.dirname(image_path),
            f"{name_without_ext}_metadata.json"
        )
        
        return FileOperations.load_json(metadata_file)
    
    @staticmethod
    def batch_process_images(input_dir, output_dir, process_func, pattern="*"):
        """
        批量处理图像
        
        参数:
            input_dir: 输入目录
            output_dir: 输出目录
            process_func: 处理函数，接收图像路径，返回处理后的图像
            pattern: 文件匹配模式
            
        返回:
            处理结果列表
        """
        if not os.path.exists(input_dir):
            return []
        
        FileOperations.ensure_directory(output_dir)
        
        image_files = FileOperations.get_image_files(input_dir)
        results = []
        
        for image_file in image_files:
            try:
                # 处理图像
                processed_image = process_func(image_file)
                
                if processed_image is not None:
                    # 生成输出文件名
                    base_name = os.path.basename(image_file)
                    output_path = os.path.join(output_dir, base_name)
                    
                    # 保存处理后的图像
                    from utils.image_loader import ImageLoader
                    ImageLoader.save_image(processed_image, output_path)
                    
                    results.append({
                        'input': image_file,
                        'output': output_path,
                        'success': True
                    })
                else:
                    results.append({
                        'input': image_file,
                        'output': None,
                        'success': False,
                        'error': '处理函数返回None'
                    })
                    
            except Exception as e:
                results.append({
                    'input': image_file,
                    'output': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results


# 模块级别的便捷函数

def ensure_directory(directory_path):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory_path: 目录路径
    """
    os.makedirs(directory_path, exist_ok=True)