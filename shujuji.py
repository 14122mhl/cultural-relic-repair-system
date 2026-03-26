import sys
import os
import random
import time
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QProgressBar, QListWidget, QGroupBox,
                             QRadioButton, QCheckBox, QSlider, QSpinBox, QComboBox,
                             QGridLayout, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QColor

import numpy as np
import cv2
from pathlib import Path


class D(QWidget):
    """数据集生成页面（整合高级污渍生成算法）"""
    back_to_initial = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.original_image = None  # 原始图像
        self.processed_image = None  # 处理后的图像
        self.image_path = ""  # 当前处理的图像路径
        self.image_paths = []  # 批量处理的图像路径列表
        self.output_dir = ""  # 输出目录
        self.processing_thread = None  # 处理线程
        self.is_processing = False  # 是否正在处理

        self.init_ui()

    def init_ui(self):
        """初始化用户界面（按算法参数调整功能选择区）"""
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 左侧控制面板（优化布局，按算法分类参数）
        left_panel = QWidget()
        left_panel.setMinimumWidth(320)
        left_panel.setMaximumWidth(350)  # 限制宽度减少空白
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(12)

        # 标题和返回按钮
        top_layout = QHBoxLayout()
        title_label = QLabel("数据集生成")
        title_font = QFont("SimHei", 16, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1a365d;")
        top_layout.addWidget(title_label)

        top_layout.addStretch()
        back_btn = QPushButton("返回首页")
        back_btn.setFont(QFont("SimHei", 10))
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        back_btn.clicked.connect(self.back_to_initial.emit)
        top_layout.addWidget(back_btn)
        left_layout.addLayout(top_layout)

        # 1. 图像选择区域（保留单张/批量功能）
        select_group = QGroupBox("图像选择")
        select_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        select_layout = QVBoxLayout(select_group)
        select_layout.setContentsMargins(8, 8, 8, 8)
        select_layout.setSpacing(8)

        # 单张/批量选择
        radio_layout = QHBoxLayout()
        self.single_radio = QRadioButton("单张处理")
        self.batch_radio = QRadioButton("批量处理")
        self.single_radio.setChecked(True)
        self.single_radio.setFont(QFont("SimHei", 10))
        self.batch_radio.setFont(QFont("SimHei", 10))
        radio_layout.addWidget(self.single_radio)
        radio_layout.addWidget(self.batch_radio)
        select_layout.addLayout(radio_layout)

        # 选择按钮
        self.select_btn = QPushButton("选择图像/文件夹")
        self.select_btn.setFont(QFont("SimHei", 10))
        self.select_btn.setMinimumHeight(30)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0f2fe;
                color: #1a365d;
                border: 1px solid #93c5fd;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #bfdbfe;
            }
        """)
        self.select_btn.clicked.connect(self.select_images)
        select_layout.addWidget(self.select_btn)

        # 显示选择的路径
        self.path_label = QLabel("未选择任何文件")
        self.path_label.setFont(QFont("SimHei", 9))
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #4a5568; border: 1px solid #e2e8f0; padding: 5px; border-radius: 4px;")
        select_layout.addWidget(self.path_label)

        left_layout.addWidget(select_group)

        # 2. 损失类型选择（按算法分为两种污渍类型）
        damage_group = QGroupBox("损失类型（污渍生成）")
        damage_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        damage_layout = QVBoxLayout(damage_group)
        damage_layout.setContentsMargins(8, 8, 8, 8)
        damage_layout.setSpacing(10)

        # 2.1 基础污渍（文档1算法：浅褐色/灰色自然污渍）
        self.basic_stain_check = QCheckBox("基础自然污渍（模拟纸张老化）")
        self.basic_stain_check.setChecked(True)
        self.basic_stain_check.setFont(QFont("SimHei", 10))
        damage_layout.addWidget(self.basic_stain_check)

        # 基础污渍参数（对应create_stain_params）
        basic_params = QWidget()
        basic_params.setStyleSheet("margin-left: 15px;")
        basic_params_layout = QGridLayout(basic_params)
        basic_params_layout.setSpacing(8)

        # 污渍数量
        basic_params_layout.addWidget(QLabel("污渍数量:", font=QFont("SimHei", 9)), 0, 0)
        self.basic_stain_count = QSpinBox()
        self.basic_stain_count.setRange(1, 20)
        self.basic_stain_count.setValue(5)  # 默认5个污渍
        basic_params_layout.addWidget(self.basic_stain_count, 0, 1)

        # 最大尺寸比例
        basic_params_layout.addWidget(QLabel("最大尺寸比例(%):", font=QFont("SimHei", 9)), 1, 0)
        self.basic_max_ratio = QSpinBox()
        self.basic_max_ratio.setRange(1, 10)
        self.basic_max_ratio.setValue(5)  # 最大5%图像尺寸
        basic_params_layout.addWidget(self.basic_max_ratio, 1, 1)

        # 最小尺寸比例
        basic_params_layout.addWidget(QLabel("最小尺寸比例(%):", font=QFont("SimHei", 9)), 2, 0)
        self.basic_min_ratio = QSpinBox()
        self.basic_min_ratio.setRange(1, 5)
        self.basic_min_ratio.setValue(1)  # 最小1%图像尺寸
        basic_params_layout.addWidget(self.basic_min_ratio, 2, 1)

        # 边缘偏向概率
        basic_params_layout.addWidget(QLabel("边缘偏向概率(%):", font=QFont("SimHei", 9)), 3, 0)
        self.basic_edge_bias = QSpinBox()
        self.basic_edge_bias.setRange(0, 100)
        self.basic_edge_bias.setValue(30)  # 30%概率生成在边缘
        basic_params_layout.addWidget(self.basic_edge_bias, 3, 1)

        damage_layout.addWidget(basic_params)

        # 2.2 沙漠风化污渍（文档2算法：裂纹/腐蚀/沙蚀）
        self.advanced_stain_check = QCheckBox("沙漠风化污渍（裂纹/腐蚀/沙蚀）")
        self.advanced_stain_check.setChecked(False)
        self.advanced_stain_check.setFont(QFont("SimHei", 10))
        damage_layout.addWidget(self.advanced_stain_check)

        # 高级污渍参数（对应create_advanced_stain_params）
        advanced_params = QWidget()
        advanced_params.setStyleSheet("margin-left: 15px;")
        advanced_params_layout = QGridLayout(advanced_params)
        advanced_params_layout.setSpacing(8)

        # 污渍数量
        advanced_params_layout.addWidget(QLabel("污渍数量:", font=QFont("SimHei", 9)), 0, 0)
        self.advanced_stain_count = QSpinBox()
        self.advanced_stain_count.setRange(1, 15)
        self.advanced_stain_count.setValue(3)  # 默认3个风化污渍
        advanced_params_layout.addWidget(self.advanced_stain_count, 0, 1)

        # 最大尺寸比例
        advanced_params_layout.addWidget(QLabel("最大尺寸比例(%):", font=QFont("SimHei", 9)), 1, 0)
        self.advanced_max_ratio = QSpinBox()
        self.advanced_max_ratio.setRange(3, 20)
        self.advanced_max_ratio.setValue(15)  # 最大15%图像尺寸（风化污渍更大）
        advanced_params_layout.addWidget(self.advanced_max_ratio, 1, 1)

        # 最小尺寸比例
        advanced_params_layout.addWidget(QLabel("最小尺寸比例(%):", font=QFont("SimHei", 9)), 2, 0)
        self.advanced_min_ratio = QSpinBox()
        self.advanced_min_ratio.setRange(3, 10)
        self.advanced_min_ratio.setValue(3)  # 最小3%图像尺寸
        advanced_params_layout.addWidget(self.advanced_min_ratio, 2, 1)

        # 边缘偏向概率
        advanced_params_layout.addWidget(QLabel("边缘偏向概率(%):", font=QFont("SimHei", 9)), 3, 0)
        self.advanced_edge_bias = QSpinBox()
        self.advanced_edge_bias.setRange(0, 100)
        self.advanced_edge_bias.setValue(60)  # 60%概率生成在边缘（符合风化特性）
        advanced_params_layout.addWidget(self.advanced_edge_bias, 3, 1)

        # 纹理复杂度
        advanced_params_layout.addWidget(QLabel("纹理复杂度:", font=QFont("SimHei", 9)), 4, 0)
        self.advanced_complexity = QSpinBox()
        self.advanced_complexity.setRange(3, 8)
        self.advanced_complexity.setValue(5)  # 中等纹理复杂度
        advanced_params_layout.addWidget(self.advanced_complexity, 4, 1)

        damage_layout.addWidget(advanced_params)

        left_layout.addWidget(damage_group)

        # 3. 输出设置（保持原有功能）
        output_group = QGroupBox("输出设置")
        output_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.setSpacing(8)

        # 输出目录
        output_dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel("未选择目录")
        self.output_dir_label.setFont(QFont("SimHei", 9))
        self.output_dir_label.setStyleSheet(
            "color: #4a5568; border: 1px solid #e2e8f0; padding: 5px; border-radius: 4px;")
        self.output_dir_label.setWordWrap(True)

        select_dir_btn = QPushButton("浏览")
        select_dir_btn.setFont(QFont("SimHei", 9))
        select_dir_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0f2fe;
                color: #1a365d;
                border: 1px solid #93c5fd;
                border-radius: 4px;
                padding: 2px 8px;
            }
        """)
        select_dir_btn.clicked.connect(self.select_output_dir)

        output_dir_layout.addWidget(self.output_dir_label, 1)
        output_dir_layout.addWidget(select_dir_btn)
        output_layout.addLayout(output_dir_layout)

        # 保存格式
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("保存格式:", font=QFont("SimHei", 9)))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "BMP"])
        self.format_combo.setCurrentText("PNG")
        format_layout.addWidget(self.format_combo)
        output_layout.addLayout(format_layout)

        left_layout.addWidget(output_group)

        # 4. 操作按钮
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)

        self.process_btn = QPushButton("开始生成")
        self.process_btn.setFont(QFont("SimHei", 12, QFont.Weight.Bold))
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:disabled {
                background-color: #a7f3d0;
                color: #64748b;
            }
        """)
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        btn_layout.addWidget(self.process_btn)

        self.cancel_btn = QPushButton("取消生成")
        self.cancel_btn.setFont(QFont("SimHei", 10))
        self.cancel_btn.setMinimumHeight(30)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:disabled {
                background-color: #fecaca;
                color: #64748b;
            }
        """)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)

        left_layout.addLayout(btn_layout)
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # 中间图像预览区域（上下布局，保持原有体验）
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(10)

        preview_label = QLabel("图像预览（原始 vs 带污渍）")
        preview_label.setFont(QFont("SimHei", 12, QFont.Weight.Bold))
        preview_label.setStyleSheet("color: #1a365d;")
        center_layout.addWidget(preview_label)

        # 原始图像显示
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_title = QLabel("原始图像")
        original_title.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        original_title.setStyleSheet("color: #4a5568;")
        original_layout.addWidget(original_title)

        self.original_preview = QLabel("请选择图像")
        self.original_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_preview.setMinimumSize(500, 400)
        self.original_preview.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; color: #64748b;")
        original_layout.addWidget(self.original_preview)
        center_layout.addWidget(original_container)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        center_layout.addWidget(line)

        # 处理后图像显示
        processed_container = QWidget()
        processed_layout = QVBoxLayout(processed_container)
        processed_title = QLabel("带污渍图像")
        processed_title.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        processed_title.setStyleSheet("color: #4a5568;")
        processed_layout.addWidget(processed_title)

        self.processed_preview = QLabel("尚未生成")
        self.processed_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_preview.setMinimumSize(500, 400)
        self.processed_preview.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; color: #64748b;")
        processed_layout.addWidget(self.processed_preview)
        center_layout.addWidget(processed_container)

        # 预览控制
        preview_ctrl = QHBoxLayout()
        self.refresh_preview_btn = QPushButton("刷新预览")
        self.refresh_preview_btn.setFont(QFont("SimHei", 10))
        self.refresh_preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0f2fe;
                color: #1a365d;
                border: 1px solid #93c5fd;
                border-radius: 6px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #bfdbfe;
            }
            QPushButton:disabled {
                background-color: #f1f5f9;
                color: #94a3b8;
            }
        """)
        self.refresh_preview_btn.clicked.connect(self.refresh_preview)
        self.refresh_preview_btn.setEnabled(False)
        preview_ctrl.addWidget(self.refresh_preview_btn)
        preview_ctrl.addStretch()
        center_layout.addLayout(preview_ctrl)

        main_layout.addWidget(center_panel, 1)

        # 右侧进度和日志区域（保持原有功能）
        right_panel = QWidget()
        right_panel.setMinimumWidth(300)
        right_panel.setMaximumWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(12)

        # 处理进度
        progress_group = QGroupBox("处理进度")
        progress_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.status_label = QLabel("等待操作...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("SimHei", 10))
        self.status_label.setStyleSheet("color: #4a5568;")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        right_layout.addWidget(progress_group)

        # 处理日志
        log_group = QGroupBox("处理日志")
        log_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        log_layout = QVBoxLayout(log_group)
        self.log_list = QListWidget()
        self.log_list.setFont(QFont("SimHei", 9))
        log_layout.addWidget(self.log_list)
        right_layout.addWidget(log_group, 1)  # 日志占最大面积

        main_layout.addWidget(right_panel)

    # ------------------------------
    # 核心算法：文档1-基础自然污渍生成
    # ------------------------------
    def create_basic_stain_params(self, height, width):
        """生成基础污渍参数（模拟纸张老化，浅褐色/灰色污渍）"""
        stains = []
        num_stains = self.basic_stain_count.value()
        max_size_ratio = self.basic_max_ratio.value() / 100.0
        min_size_ratio = self.basic_min_ratio.value() / 100.0
        edge_bias = self.basic_edge_bias.value() / 100.0

        # 计算污渍尺寸范围
        max_size = int(max(height, width) * max_size_ratio)
        min_size = int(max(height, width) * min_size_ratio)
        min_size = max(1, min_size)

        for _ in range(num_stains):
            # 1. 位置选择（倾向边缘）
            if random.random() < edge_bias:
                edge = random.choice(['top', 'bottom', 'left', 'right'])
                if edge == 'top':
                    center_y = random.randint(0, int(height * 0.1))
                    center_x = random.randint(0, width)
                elif edge == 'bottom':
                    center_y = random.randint(int(height * 0.9), height)
                    center_x = random.randint(0, width)
                elif edge == 'left':
                    center_x = random.randint(0, int(width * 0.1))
                    center_y = random.randint(0, height)
                else:  # right
                    center_x = random.randint(int(width * 0.9), width)
                    center_y = random.randint(0, height)
            else:
                center_x = random.randint(0, width)
                center_y = random.randint(0, height)

            # 2. 尺寸选择
            size_x = random.randint(min_size, max_size)
            size_y = random.randint(min_size, max_size)

            # 3. 颜色选择（浅褐色，BGR格式）
            color = (
                random.randint(80, 120),  # 蓝色通道
                random.randint(70, 110),  # 绿色通道
                random.randint(60, 100)  # 红色通道
            )

            # 4. 透明度（20%-50%）
            alpha = random.uniform(0.2, 0.5)

            # 5. 模糊半径（1-3像素）
            blur = random.randint(1, 3)

            stains.append({
                'center': (center_x, center_y),
                'size': (size_x, size_y),
                'color': color,
                'alpha': alpha,
                'blur': blur
            })
        return stains

    def apply_basic_stains(self, image):
        """应用基础污渍到图像"""
        if not self.basic_stain_check.isChecked():
            return image

        stained_image = image.copy()
        height, width = stained_image.shape[:2]
        stains = self.create_basic_stain_params(height, width)

        for stain in stains:
            # 生成椭圆污渍掩码
            mask = np.zeros((height, width), dtype=np.uint8)
            center_x, center_y = stain['center']
            size_x, size_y = stain['size']
            axes = (size_x // 2, size_y // 2)
            cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)

            # 模糊掩码边缘
            kernel_size = stain['blur'] * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

            # 生成半透明污渍层
            mask_float = mask.astype(np.float32) / 255.0
            alpha_map = mask_float * stain['alpha']
            stain_layer = np.zeros_like(stained_image)
            stain_layer[:, :] = stain['color']

            # 混合到原图
            for c in range(3):
                stained_image[:, :, c] = (
                        stained_image[:, :, c] * (1 - alpha_map) +
                        stain_layer[:, :, c] * alpha_map
                )

        return np.clip(stained_image, 0, 255).astype(np.uint8)

    # ------------------------------
    # 核心算法：文档2-沙漠风化污渍生成
    # ------------------------------
    def create_advanced_stain_params(self, height, width):
        """生成高级风化污渍参数（裂纹/腐蚀/沙蚀）"""
        stains = []
        num_stains = self.advanced_stain_count.value()
        max_size_ratio = self.advanced_max_ratio.value() / 100.0
        min_size_ratio = self.advanced_min_ratio.value() / 100.0
        edge_bias = self.advanced_edge_bias.value() / 100.0
        complexity = self.advanced_complexity.value()

        # 计算污渍尺寸范围
        max_size = int(max(height, width) * max_size_ratio)
        min_size = int(max(height, width) * min_size_ratio)
        min_size = max(5, min_size)

        # 污渍类型列表
        stain_types = ['crack', 'erosion', 'sand_wear', 'stain']

        for _ in range(num_stains):
            # 1. 位置选择（高概率边缘/角落）
            if random.random() < edge_bias:
                edge = random.choice(['top', 'bottom', 'left', 'right', 'corner'])
                if edge == 'top':
                    center_y = random.randint(0, int(height * 0.15))
                    center_x = random.randint(0, width)
                elif edge == 'bottom':
                    center_y = random.randint(int(height * 0.85), height)
                    center_x = random.randint(0, width)
                elif edge == 'left':
                    center_x = random.randint(0, int(width * 0.15))
                    center_y = random.randint(0, height)
                elif edge == 'right':
                    center_x = random.randint(int(width * 0.85), width)
                    center_y = random.randint(0, height)
                else:  # corner
                    corners = [
                        (int(width * 0.1), int(height * 0.1)),
                        (int(width * 0.9), int(height * 0.1)),
                        (int(width * 0.1), int(height * 0.9)),
                        (int(width * 0.9), int(height * 0.9))
                    ]
                    center_x, center_y = random.choice(corners)
            else:
                center_x = random.randint(0, width)
                center_y = random.randint(0, height)

            # 2. 尺寸选择（不规则）
            size_x = random.randint(min_size, max_size)
            size_y = random.randint(min_size, max_size)

            # 3. 污渍类型与参数
            stain_type = random.choice(stain_types)
            if stain_type == 'crack':
                texture_intensity = random.uniform(0.8, 1.0)
                color = (random.randint(40, 70), random.randint(30, 60), random.randint(25, 55))  # 深棕色
                alpha = random.uniform(0.7, 0.9)  # 裂纹更明显
            elif stain_type == 'erosion':
                texture_intensity = random.uniform(0.7, 0.9)
                color = (random.randint(80, 110), random.randint(70, 100), random.randint(60, 90))  # 沙黄色
                alpha = random.uniform(0.3, 0.6)
            elif stain_type == 'sand_wear':
                texture_intensity = random.uniform(0.6, 0.8)
                color = (random.randint(100, 130), random.randint(90, 120), random.randint(80, 110))  # 浅沙色
                alpha = random.uniform(0.3, 0.6)
            else:  # stain
                texture_intensity = random.uniform(0.4, 0.7)
                color = (random.randint(70, 100), random.randint(60, 90), random.randint(50, 80))  # 褐色
                alpha = random.uniform(0.3, 0.6)

            # 4. 其他参数
            blur = random.randint(1, 3)

            stains.append({
                'center': (center_x, center_y),
                'size': (size_x, size_y),
                'type': stain_type,
                'color': color,
                'alpha': alpha,
                'blur': blur,
                'texture_intensity': texture_intensity,
                'complexity': complexity
            })
        return stains

    def generate_texture_mask(self, stain, height, width):
        """生成风化污渍的纹理掩码（裂纹分支/腐蚀坑洼/沙蚀条纹）"""
        center_x, center_y = stain['center']
        size_x, size_y = stain['size']
        stain_type = stain['type']
        texture_intensity = stain['texture_intensity']
        complexity = stain['complexity']

        # 基础掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        if stain_type == 'crack':
            # 裂纹：细长椭圆
            axes = (int(size_x * 0.8), int(size_y * 0.1))
            angle = random.randint(0, 180)
            cv2.ellipse(mask, (center_x, center_y), axes, angle, 0, 360, 255, -1)
        else:
            # 其他类型：常规椭圆
            axes = (size_x // 2, size_y // 2)
            cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)

        # 纹理掩码
        texture_mask = np.zeros((height, width), dtype=np.uint8)
        if stain_type == 'crack':
            # 裂纹分支
            for _ in range(complexity):
                length = random.randint(int(size_x * 0.3), size_x)
                thickness = random.randint(1, max(2, int(size_x * 0.05)))
                angle = random.randint(0, 360)
                end_x = int(center_x + length * np.cos(np.radians(angle)))
                end_y = int(center_y + length * np.sin(np.radians(angle)))
                cv2.line(texture_mask, (center_x, center_y), (end_x, end_y), 255, thickness)

                # 小分支
                if random.random() > 0.7:
                    branch_length = random.randint(int(length * 0.2), int(length * 0.5))
                    branch_angle = angle + random.randint(-45, 45)
                    branch_x = int(center_x + length * 0.5 * np.cos(np.radians(angle)))
                    branch_y = int(center_y + length * 0.5 * np.sin(np.radians(angle)))
                    branch_end_x = int(branch_x + branch_length * np.cos(np.radians(branch_angle)))
                    branch_end_y = int(branch_y + branch_length * np.sin(np.radians(branch_angle)))
                    cv2.line(texture_mask, (branch_x, branch_y), (branch_end_x, branch_end_y), 255,
                             max(1, thickness - 1))

            # 腐蚀边缘
            kernel = np.ones((random.randint(1, 3), random.randint(1, 3)), np.uint8)
            texture_mask = cv2.erode(texture_mask, kernel, iterations=1)

        elif stain_type == 'erosion':
            # 腐蚀坑洼
            for _ in range(complexity * 5):
                x = random.randint(max(0, center_x - size_x // 2), min(width - 1, center_x + size_x // 2))
                y = random.randint(max(0, center_y - size_y // 2), min(height - 1, center_y + size_y // 2))
                if mask[y, x] > 0:
                    radius = random.randint(1, max(2, int(min(size_x, size_y) * 0.1)))
                    cv2.circle(texture_mask, (x, y), radius, 255, -1)

            # 沙子纹理
            sand_mask = np.zeros((height, width), dtype=np.uint8)
            for _ in range(complexity * 10):
                x = random.randint(max(0, center_x - size_x // 2), min(width - 1, center_x + size_x // 2))
                y = random.randint(max(0, center_y - size_y // 2), min(height - 1, center_y + size_y // 2))
                if mask[y, x] > 0:
                    radius = random.randint(1, 2)
                    cv2.circle(sand_mask, (x, y), radius, 255, -1)
            texture_mask = cv2.add(texture_mask, sand_mask)

        elif stain_type == 'sand_wear':
            # 沙蚀条纹
            for i in range(complexity):
                angle = random.randint(0, 180)
                thickness = random.randint(1, max(2, int(min(size_x, size_y) * 0.05)))
                start_x = center_x - size_x // 2
                end_x = center_x + size_x // 2
                start_y = center_y - size_y // 2 + int(size_y * i / complexity)
                end_y = start_y
                cv2.line(texture_mask, (start_x, start_y), (end_x, end_y), 255, thickness)

        else:
            # 普通污渍颗粒
            for _ in range(complexity * 8):
                x = random.randint(max(0, center_x - size_x // 2), min(width - 1, center_x + size_x // 2))
                y = random.randint(max(0, center_y - size_y // 2), min(height - 1, center_y + size_y // 2))
                if mask[y, x] > 0:
                    radius = random.randint(1, max(2, int(min(size_x, size_y) * 0.03)))
                    cv2.circle(texture_mask, (x, y), radius, 255, -1)

        # 合并掩码与纹理
        if stain_type != 'crack':
            mask = cv2.add(mask, texture_mask)
        else:
            mask = texture_mask

        # 调整纹理强度与模糊
        mask = cv2.addWeighted(mask, texture_intensity, np.zeros_like(mask), 1 - texture_intensity, 0)
        kernel_size = stain['blur'] * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        return mask

    def apply_advanced_stains(self, image):
        """应用沙漠风化污渍到图像"""
        if not self.advanced_stain_check.isChecked():
            return image

        stained_image = image.copy()
        height, width = stained_image.shape[:2]
        stains = self.create_advanced_stain_params(height, width)

        for stain in stains:
            # 生成纹理掩码
            mask = self.generate_texture_mask(stain, height, width)

            # 生成半透明污渍层
            mask_float = mask.astype(np.float32) / 255.0
            alpha_map = mask_float * stain['alpha']
            stain_layer = np.zeros_like(stained_image)
            stain_layer[:, :] = stain['color']

            # 混合到原图
            for c in range(3):
                stained_image[:, :, c] = (
                        stained_image[:, :, c] * (1 - alpha_map) +
                        stain_layer[:, :, c] * alpha_map
                )

        return np.clip(stained_image, 0, 255).astype(np.uint8)

    # ------------------------------
    # 界面交互与工具函数
    # ------------------------------
    def select_images(self):
        """选择单张/批量图像（保持原有逻辑）"""
        if self.single_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
            )
            if file_path:
                self.image_paths = [file_path]
                self.path_label.setText(f"已选择: {os.path.basename(file_path)}")
                self.load_image(file_path)
        else:
            dir_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
            if dir_path:
                image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
                self.image_paths = [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if Path(f).suffix.lower() in image_extensions
                ]
                if self.image_paths:
                    self.path_label.setText(f"已选择文件夹: {dir_path}\n包含 {len(self.image_paths)} 张图像")
                    self.load_image(self.image_paths[0])
                else:
                    QMessageBox.warning(self, "警告", "所选文件夹中无图像文件")
                    return

        # 更新按钮状态
        self.process_btn.setEnabled(True if self.output_dir else False)
        self.refresh_preview_btn.setEnabled(True)

    def load_image(self, file_path):
        """加载图像并显示预览"""
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise Exception("无法读取图像")
            self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.display_image(self.original_preview, self.original_image)
            self.refresh_preview()
            self.log_message(f"加载图像成功: {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")
            self.log_message(f"加载失败: {str(e)}")

    def display_image(self, label, image_array):
        """在QLabel中显示图像"""
        if image_array is None:
            label.setText("无图像")
            return
        height, width, channels = image_array.shape
        bytes_per_line = channels * width
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        scaled_image = q_image.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(QPixmap.fromImage(scaled_image))
        label.setText("")

    def select_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
            self.process_btn.setEnabled(True if self.image_paths else False)

    def refresh_preview(self):
        """刷新预览图像（应用选中的污渍算法）"""
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()
        # 先应用基础污渍，再应用高级污渍（可叠加）
        self.processed_image = self.apply_basic_stains(self.processed_image)
        self.processed_image = self.apply_advanced_stains(self.processed_image)
        self.display_image(self.processed_preview, self.processed_image)
        self.log_message("预览更新完成（应用选中的污渍类型）")

    def start_processing(self):
        """开始批量生成（多线程处理）"""
        if not self.image_paths or not self.output_dir:
            QMessageBox.warning(self, "警告", "请选择图像和输出目录")
            return
        if not (self.basic_stain_check.isChecked() or self.advanced_stain_check.isChecked()):
            QMessageBox.warning(self, "警告", "请至少选择一种污渍类型")
            return

        # 禁用控件
        self.process_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.refresh_preview_btn.setEnabled(False)

        # 启动线程
        self.processing_thread = StainProcessingThread(
            self.image_paths, self.output_dir,
            self.apply_basic_stains, self.apply_advanced_stains,
            self.format_combo.currentText()
        )
        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.status_updated.connect(self.status_label.setText)
        self.processing_thread.log_updated.connect(self.log_message)
        self.processing_thread.finished.connect(self.processing_finished)
        self.is_processing = True
        self.processing_thread.start()

    def cancel_processing(self):
        """取消处理"""
        if self.processing_thread and self.is_processing:
            self.processing_thread.cancel()
            self.status_label.setText("正在取消...")
            self.log_message("用户取消生成")

    def processing_finished(self, success):
        """处理完成回调"""
        self.is_processing = False
        self.cancel_btn.setEnabled(False)
        self.select_btn.setEnabled(True)
        self.refresh_preview_btn.setEnabled(True)
        self.process_btn.setEnabled(True if self.image_paths else False)
        if success:
            QMessageBox.information(self, "完成", "所有图像生成完成")
            self.status_label.setText("处理完成")
        else:
            self.status_label.setText("处理取消")

    def log_message(self, message):
        """添加日志"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_list.addItem(f"[{timestamp}] {message}")
        self.log_list.scrollToBottom()


class StainProcessingThread(QThread):
    """污渍生成线程（避免界面卡顿）"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def __init__(self, image_paths, output_dir, basic_stain_func, advanced_stain_func, output_format):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.basic_stain_func = basic_stain_func
        self.advanced_stain_func = advanced_stain_func
        self.output_format = output_format.lower()
        self.canceled = False

    def run(self):
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            if self.canceled:
                self.finished.emit(False)
                return

            try:
                # 读取图像
                image = cv2.imread(path)
                if image is None:
                    self.log_updated.emit(f"跳过: 无法读取 {os.path.basename(path)}")
                    continue
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 应用污渍算法
                processed_image = self.basic_stain_func(rgb_image)
                processed_image = self.advanced_stain_func(processed_image)

                # 保存图像
                filename = Path(path).stem
                output_path = os.path.join(self.output_dir, f"{filename}_stained.{self.output_format}")
                bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, bgr_image)

                # 更新进度
                progress = int((i + 1) / total * 100)
                self.progress_updated.emit(progress)
                self.status_updated.emit(f"处理 {i + 1}/{total}")
                self.log_updated.emit(f"生成成功: {os.path.basename(output_path)}")

            except Exception as e:
                self.log_updated.emit(f"处理失败 {os.path.basename(path)}: {str(e)}")

        self.progress_updated.emit(100)
        self.finished.emit(not self.canceled)

    def cancel(self):
        self.canceled = True


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setFont(QFont("SimHei"))  # 确保中文显示
    window = DatasetGeneratorPage()
    window.setWindowTitle("数据集生成（污渍算法版）")
    window.resize(1400, 900)
    window.show()
    sys.exit(app.exec())