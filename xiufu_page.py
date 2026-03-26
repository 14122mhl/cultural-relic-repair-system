import sys
import os
import time
import cv2
import numpy as np
import glob
from math import cos, sin
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QProgressBar, QListWidget, QGroupBox,
                             QRadioButton, QSlider, QComboBox, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage


class RelicRestorationPage(QWidget):
    """文物修复页面（最终优化版）"""
    back_to_initial = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.original_images = []  # 原始图像列表
        self.restored_images = []  # 修复后的图像列表
        self.image_paths = []  # 图像路径列表
        self.current_index = 0  # 当前显示的图像索引
        self.output_dir = ""  # 输出目录
        self.processing_thread = None  # 处理线程
        self.is_processing = False  # 是否正在处理
        self.selected_algo = 0  # 选中的算法索引（0:快速,1:精细,2:深度）
        
        # 标注功能相关变量
        self.annotation_image = None
        self.annotation_mask = None
        self.annotation_enabled = False
        self.temp_mask_dir = "D:/python_chuangxin/jiemianpythonProject/V2/zanshibaocun"
        self.painting = False
        
        # 确保临时目录存在
        os.makedirs(self.temp_mask_dir, exist_ok=True)

        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setMinimumWidth(280)
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(12)

        # 标题和返回按钮
        top_layout = QHBoxLayout()
        top_layout.setSpacing(5)

        title_label = QLabel("文物图像修复")
        title_font = QFont("SimHei", 14, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #1a365d;")
        top_layout.addWidget(title_label)

        top_layout.addStretch()

        back_btn = QPushButton("返回首页")
        back_btn.setFont(QFont("SimHei", 9))
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        back_btn.setMinimumHeight(25)
        back_btn.clicked.connect(self.back_to_initial.emit)
        top_layout.addWidget(back_btn)

        left_layout.addLayout(top_layout)

        # 1. 图像选择区域 (支持单张/批量)
        select_group = QGroupBox("图像选择")
        select_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        select_layout = QVBoxLayout(select_group)
        select_layout.setContentsMargins(8, 8, 8, 8)
        select_layout.setSpacing(8)

        # 单张/批量选择
        radio_layout = QHBoxLayout()
        self.single_radio = QRadioButton("单张处理")
        self.batch_radio = QRadioButton("批量处理")
        self.single_radio.setChecked(True)
        self.single_radio.setFont(QFont("SimHei", 9))
        self.batch_radio.setFont(QFont("SimHei", 9))
        radio_layout.addWidget(self.single_radio)
        radio_layout.addWidget(self.batch_radio)
        select_layout.addLayout(radio_layout)

        # 选择按钮
        self.select_btn = QPushButton("选择图像/文件夹")
        self.select_btn.setFont(QFont("SimHei", 9))
        self.select_btn.setMinimumHeight(28)
        self.select_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0f2fe;
                color: #1a365d;
                border: 1px solid #93c5fd;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #bfdbfe;
            }
        """)
        self.select_btn.clicked.connect(self.select_images)
        select_layout.addWidget(self.select_btn)

        # 显示选择的路径/数量
        self.path_label = QLabel("未选择任何文件")
        self.path_label.setFont(QFont("SimHei", 8))
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #4a5568; border: 1px solid #e2e8f0; padding: 4px; border-radius: 3px;")
        select_layout.addWidget(self.path_label)

        left_layout.addWidget(select_group)

        # 2. 修复参数设置（分两块）
        # 2.1 算法选择（每个算法一行，右侧有选择按钮）
        algo_group = QGroupBox("算法选择")
        algo_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setContentsMargins(8, 8, 8, 8)
        algo_layout.setSpacing(8)

        # 快速修复
        fast_layout = QHBoxLayout()
        fast_layout.setSpacing(5)
        fast_label = QLabel("快速修复：适用于轻度损伤，处理速度快")
        fast_label.setFont(QFont("SimHei", 9))
        fast_label.setStyleSheet("color: #4a5568;")
        fast_label.setWordWrap(True)

        self.fast_btn = QPushButton("选择")
        self.fast_btn.setFont(QFont("SimHei", 8))
        self.fast_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.fast_btn.setChecked(True)
        self.fast_btn.clicked.connect(lambda: self.set_selected_algo(0))

        fast_layout.addWidget(fast_label, 1)
        fast_layout.addWidget(self.fast_btn)
        algo_layout.addLayout(fast_layout)

        # 精细修复
        fine_layout = QHBoxLayout()
        fine_layout.setSpacing(5)
        fine_label = QLabel("精细修复：保留更多细节，修复效果更好")
        fine_label.setFont(QFont("SimHei", 9))
        fine_label.setStyleSheet("color: #4a5568;")
        fine_label.setWordWrap(True)

        self.fine_btn = QPushButton("选择")
        self.fine_btn.setFont(QFont("SimHei", 8))
        self.fine_btn.setStyleSheet("""
            QPushButton {
                background-color: #94a3b8;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #64748b;
            }
        """)
        self.fine_btn.clicked.connect(lambda: self.set_selected_algo(1))

        fine_layout.addWidget(fine_label, 1)
        fine_layout.addWidget(self.fine_btn)
        algo_layout.addLayout(fine_layout)

        # 深度修复
        deep_layout = QHBoxLayout()
        deep_layout.setSpacing(5)
        deep_label = QLabel("深度修复：适用于重度损伤，处理耗时较长")
        deep_label.setFont(QFont("SimHei", 9))
        deep_label.setStyleSheet("color: #4a5568;")
        deep_label.setWordWrap(True)

        self.deep_btn = QPushButton("选择")
        self.deep_btn.setFont(QFont("SimHei", 8))
        self.deep_btn.setStyleSheet("""
            QPushButton {
                background-color: #94a3b8;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #64748b;
            }
        """)
        self.deep_btn.clicked.connect(lambda: self.set_selected_algo(2))

        deep_layout.addWidget(deep_label, 1)
        deep_layout.addWidget(self.deep_btn)
        algo_layout.addLayout(deep_layout)

        left_layout.addWidget(algo_group)

        # 2.2 强度/降噪/细节滑动调节
        param_group = QGroupBox("参数调节")
        param_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        param_layout = QVBoxLayout(param_group)
        param_layout.setContentsMargins(8, 8, 8, 8)
        param_layout.setSpacing(10)

        # 修复强度
        strength_layout = QVBoxLayout()
        strength_title = QHBoxLayout()
        strength_title.addWidget(QLabel("修复强度:", font=QFont("SimHei", 9)))
        self.strength_value = QLabel("5", font=QFont("SimHei", 9))
        strength_title.addStretch()
        strength_title.addWidget(self.strength_value)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(1, 10)
        self.strength_slider.setValue(5)
        self.strength_slider.valueChanged.connect(lambda value: self.strength_value.setText(str(value)))

        strength_layout.addLayout(strength_title)
        strength_layout.addWidget(self.strength_slider)
        param_layout.addLayout(strength_layout)

        # 降噪程度
        denoise_layout = QVBoxLayout()
        denoise_title = QHBoxLayout()
        denoise_title.addWidget(QLabel("降噪程度:", font=QFont("SimHei", 9)))
        self.denoise_value = QLabel("50%", font=QFont("SimHei", 9))
        denoise_title.addStretch()
        denoise_title.addWidget(self.denoise_value)

        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setRange(0, 100)
        self.denoise_slider.setValue(50)
        self.denoise_slider.valueChanged.connect(lambda value: self.denoise_value.setText(f"{value}%"))

        denoise_layout.addLayout(denoise_title)
        denoise_layout.addWidget(self.denoise_slider)
        param_layout.addLayout(denoise_layout)

        # 细节保留
        detail_layout = QVBoxLayout()
        detail_title = QHBoxLayout()
        detail_title.addWidget(QLabel("细节保留:", font=QFont("SimHei", 9)))
        self.detail_value = QLabel("70%", font=QFont("SimHei", 9))
        detail_title.addStretch()
        detail_title.addWidget(self.detail_value)

        self.detail_slider = QSlider(Qt.Orientation.Horizontal)
        self.detail_slider.setRange(0, 100)
        self.detail_slider.setValue(70)
        self.detail_slider.valueChanged.connect(lambda value: self.detail_value.setText(f"{value}%"))

        detail_layout.addLayout(detail_title)
        detail_layout.addWidget(self.detail_slider)
        param_layout.addLayout(detail_layout)

        left_layout.addWidget(param_group)

        # 3. 输出设置
        output_group = QGroupBox("输出设置")
        output_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.setSpacing(8)

        # 输出目录
        output_dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel("未选择目录")
        self.output_dir_label.setFont(QFont("SimHei", 8))
        self.output_dir_label.setStyleSheet(
            "color: #4a5568; border: 1px solid #e2e8f0; padding: 3px; border-radius: 3px;")
        self.output_dir_label.setWordWrap(True)
        self.output_dir_label.setMinimumHeight(22)

        select_dir_btn = QPushButton("浏览")
        select_dir_btn.setFont(QFont("SimHei", 8))
        select_dir_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0f2fe;
                color: #1a365d;
                border: 1px solid #93c5fd;
                border-radius: 3px;
                padding: 2px 6px;
            }
        """)
        select_dir_btn.setMinimumHeight(22)
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
        self.format_combo.setFont(QFont("SimHei", 9))
        self.format_combo.setMinimumHeight(22)
        format_layout.addWidget(self.format_combo)
        output_layout.addLayout(format_layout)

        left_layout.addWidget(output_group)

        # 4. 操作按钮区域
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)

        # 处理按钮
        self.process_btn = QPushButton("开始修复")
        self.process_btn.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        self.process_btn.setMinimumHeight(32)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 5px;
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

        # 取消按钮
        self.cancel_btn = QPushButton("取消修复")
        self.cancel_btn.setFont(QFont("SimHei", 9))
        self.cancel_btn.setMinimumHeight(28)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 5px;
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

        # 保存按钮
        self.save_single_btn = QPushButton("保存当前结果")
        self.save_single_btn.setFont(QFont("SimHei", 9))
        self.save_single_btn.setMinimumHeight(28)
        self.save_single_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #93c5fd;
                color: #64748b;
            }
        """)
        self.save_single_btn.clicked.connect(self.save_current_restored_image)
        self.save_single_btn.setEnabled(False)
        btn_layout.addWidget(self.save_single_btn)

        self.save_batch_btn = QPushButton("批量保存结果")
        self.save_batch_btn.setFont(QFont("SimHei", 9))
        self.save_batch_btn.setMinimumHeight(28)
        self.save_batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7c3aed;
            }
            QPushButton:disabled {
                background-color: #c4b5fd;
                color: #64748b;
            }
        """)
        self.save_batch_btn.clicked.connect(self.batch_save_restored_images)
        self.save_batch_btn.setEnabled(False)
        btn_layout.addWidget(self.save_batch_btn)

        left_layout.addLayout(btn_layout)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # 中间图像预览区域
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(10)

        # 顶部：标题和批量导航
        top_center_layout = QHBoxLayout()

        # 标题
        preview_label = QLabel("图像修复前后对比")
        preview_label.setFont(QFont("SimHei", 12, QFont.Weight.Bold))
        preview_label.setStyleSheet("color: #1a365d;")
        top_center_layout.addWidget(preview_label)

        # 批量导航
        self.nav_widget = QWidget()
        nav_layout = QHBoxLayout(self.nav_widget)
        nav_layout.setSpacing(8)

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.setFont(QFont("SimHei", 9))
        self.prev_btn.setMinimumHeight(26)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #e2e8f0;
                color: #1e293b;
                border: none;
                border-radius: 3px;
                padding: 3px 10px;
            }
            QPushButton:hover {
                background-color: #cbd5e1;
            }
            QPushButton:disabled {
                background-color: #f1f5f9;
                color: #94a3b8;
            }
        """)
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.prev_btn.setEnabled(False)

        self.image_index_label = QLabel("")
        self.image_index_label.setFont(QFont("SimHei", 9))

        self.next_btn = QPushButton("下一张")
        self.next_btn.setFont(QFont("SimHei", 9))
        self.next_btn.setMinimumHeight(26)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #e2e8f0;
                color: #1e293b;
                border: none;
                border-radius: 3px;
                padding: 3px 10px;
            }
            QPushButton:hover {
                background-color: #cbd5e1;
            }
            QPushButton:disabled {
                background-color: #f1f5f9;
                color: #94a3b8;
            }
        """)
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.image_index_label)
        nav_layout.addWidget(self.next_btn)

        top_center_layout.addWidget(self.nav_widget)
        self.nav_widget.setVisible(False)

        top_center_layout.addStretch()
        center_layout.addLayout(top_center_layout)

        # 图像显示区域（上下分割）
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        preview_layout.setSpacing(8)

        # 原图显示
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.setContentsMargins(0, 0, 0, 0)
        original_layout.setSpacing(3)

        original_title = QLabel("修复前图像")
        original_title.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        original_title.setStyleSheet("color: #4a5568;")
        original_layout.addWidget(original_title)

        self.original_preview = QLabel("请选择图像")
        self.original_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_preview.setMinimumSize(550, 380)
        self.original_preview.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; color: #64748b;")
        original_layout.addWidget(self.original_preview)

        preview_layout.addWidget(original_container)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        preview_layout.addWidget(line)

        # 修复后图像显示
        restored_container = QWidget()
        restored_layout = QVBoxLayout(restored_container)
        restored_layout.setContentsMargins(0, 0, 0, 0)
        restored_layout.setSpacing(3)

        restored_title = QLabel("修复后图像")
        restored_title.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        restored_title.setStyleSheet("color: #4a5568;")
        restored_layout.addWidget(restored_title)

        self.restored_preview = QLabel("尚未修复")
        self.restored_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.restored_preview.setMinimumSize(550, 380)
        self.restored_preview.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; color: #64748b;")
        restored_layout.addWidget(self.restored_preview)

        preview_layout.addWidget(restored_container)

        center_layout.addWidget(preview_widget, 1)

        # 状态标签
        self.status_label = QLabel("等待操作...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("SimHei", 10))
        self.status_label.setStyleSheet("color: #4a5568;")
        center_layout.addWidget(self.status_label)

        main_layout.addWidget(center_panel, 1)

        # 右侧面板（从上到下：修复进度→修复统计→处理日志）
        right_panel = QWidget()
        right_panel.setMinimumWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(10)

        # 1. 修复进度（最上方）
        progress_group = QGroupBox("修复进度")
        progress_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(8, 8, 8, 8)
        progress_layout.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)

        self.progress_label = QLabel("等待开始处理...")
        self.progress_label.setFont(QFont("SimHei", 9))
        self.progress_label.setStyleSheet("color: #4a5568;")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        right_layout.addWidget(progress_group)

        # 2. 修复统计
        stats_group = QGroupBox("修复统计")
        stats_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_layout.setSpacing(6)

        self.stats_label = QLabel("""
            修复耗时: --:--:--
            修复图像: 0/0
            平均区域: -- 像素
        """)
        self.stats_label.setFont(QFont("SimHei", 9))
        self.stats_label.setStyleSheet("white-space: pre; color: #4a5568;")
        stats_layout.addWidget(self.stats_label)

        right_layout.addWidget(stats_group)

        # 手动污渍标注工作区
        annotation_group = QGroupBox("手动污渍标注")
        annotation_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        annotation_layout = QVBoxLayout(annotation_group)
        annotation_layout.setContentsMargins(8, 8, 8, 8)
        annotation_layout.setSpacing(6)

        # 标注控制按钮
        annotation_buttons_layout = QHBoxLayout()
        annotation_buttons_layout.setSpacing(4)

        self.upload_annotation_btn = QPushButton("导入图像")
        self.upload_annotation_btn.setFont(QFont("SimHei", 8))
        self.upload_annotation_btn.setMinimumHeight(25)
        self.upload_annotation_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0f2fe;
                color: #1a365d;
                border: 1px solid #93c5fd;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #bfdbfe;
            }
            QPushButton:disabled {
                background-color: #f1f5f9;
                color: #94a3b8;
                border: 1px solid #e2e8f0;
            }
        """)
        self.upload_annotation_btn.clicked.connect(self.uploadAnnotationImage)
        self.upload_annotation_btn.setEnabled(False)

        self.start_annotation_btn = QPushButton("开始标注")
        self.start_annotation_btn.setFont(QFont("SimHei", 8))
        self.start_annotation_btn.setMinimumHeight(25)
        self.start_annotation_btn.setStyleSheet("""
            QPushButton {
                background-color: #dcfce7;
                color: #166534;
                border: 1px solid #86efac;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #bbf7d0;
            }
            QPushButton:disabled {
                background-color: #f1f5f9;
                color: #94a3b8;
                border: 1px solid #e2e8f0;
            }
        """)
        self.start_annotation_btn.clicked.connect(self.startAnnotation)
        self.start_annotation_btn.setEnabled(False)

        self.confirm_annotation_btn = QPushButton("确认标注")
        self.confirm_annotation_btn.setFont(QFont("SimHei", 8))
        self.confirm_annotation_btn.setMinimumHeight(25)
        self.confirm_annotation_btn.setStyleSheet("""
            QPushButton {
                background-color: #fef3c7;
                color: #92400e;
                border: 1px solid #fcd34d;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #fde68a;
            }
            QPushButton:disabled {
                background-color: #f1f5f9;
                color: #94a3b8;
                border: 1px solid #e2e8f0;
            }
        """)
        self.confirm_annotation_btn.clicked.connect(self.confirmAnnotation)
        self.confirm_annotation_btn.setEnabled(False)

        self.reset_annotation_btn = QPushButton("重置标注")
        self.reset_annotation_btn.setFont(QFont("SimHei", 8))
        self.reset_annotation_btn.setMinimumHeight(25)
        self.reset_annotation_btn.setStyleSheet("""
            QPushButton {
                background-color: #fee2e2;
                color: #991b1b;
                border: 1px solid #fca5a5;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #fecaca;
            }
            QPushButton:disabled {
                background-color: #f1f5f9;
                color: #94a3b8;
                border: 1px solid #e2e8f0;
            }
        """)
        self.reset_annotation_btn.clicked.connect(self.resetAnnotation)
        self.reset_annotation_btn.setEnabled(False)

        annotation_buttons_layout.addWidget(self.upload_annotation_btn)
        annotation_buttons_layout.addWidget(self.start_annotation_btn)
        annotation_buttons_layout.addWidget(self.confirm_annotation_btn)
        annotation_buttons_layout.addWidget(self.reset_annotation_btn)

        # 画笔设置
        brush_settings_layout = QHBoxLayout()
        brush_settings_layout.setSpacing(4)

        self.brush_size_label = QLabel("画笔大小:")
        self.brush_size_label.setFont(QFont("SimHei", 8))
        self.brush_size_label.setStyleSheet("color: #4a5568;")

        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(5)
        self.brush_size_slider.valueChanged.connect(self.updateBrushSize)
        self.brush_size_slider.setMinimumHeight(20)

        self.brush_size_value = QLabel("5")
        self.brush_size_value.setFont(QFont("SimHei", 8))
        self.brush_size_value.setStyleSheet("color: #4a5568; min-width: 20px;")

        brush_settings_layout.addWidget(self.brush_size_label)
        brush_settings_layout.addWidget(self.brush_size_slider, 1)
        brush_settings_layout.addWidget(self.brush_size_value)

        # 标注状态显示
        self.annotation_status_label = QLabel("状态: 未导入图像")
        self.annotation_status_label.setFont(QFont("SimHei", 8, QFont.Weight.Bold))
        self.annotation_status_label.setStyleSheet("color: #94a3b8;")

        # 标注图像显示区域
        self.annotation_display = QLabel()
        self.annotation_display.setMinimumSize(280, 150)
        self.annotation_display.setMaximumHeight(180)
        self.annotation_display.setStyleSheet("""
            QLabel {
                border: 2px solid #e2e8f0;
                background-color: #f8fafc;
                border-radius: 4px;
            }
        """)
        self.annotation_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.annotation_display.setText("标注图像预览区域")
        self.annotation_display.setFont(QFont("SimHei", 9))
        self.annotation_display.setStyleSheet("""
            QLabel {
                border: 2px solid #e2e8f0;
                background-color: #f8fafc;
                border-radius: 4px;
                color: #94a3b8;
            }
        """)

        # 添加到布局
        annotation_layout.addLayout(annotation_buttons_layout)
        annotation_layout.addLayout(brush_settings_layout)
        annotation_layout.addWidget(self.annotation_status_label)
        annotation_layout.addWidget(self.annotation_display)

        right_layout.addWidget(annotation_group)

        # 3. 处理日志（占据最大面积）
        log_group = QGroupBox("处理日志")
        log_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)

        self.log_list = QListWidget()
        self.log_list.setFont(QFont("SimHei", 9))
        log_layout.addWidget(self.log_list)

        right_layout.addWidget(log_group, 1)  # 占满剩余空间

        main_layout.addWidget(right_panel)

        # 设置窗口属性
        self.setWindowTitle("文物图像修复")
        self.resize(1450, 900)

    def set_selected_algo(self, index):
        """设置选中的算法并更新按钮样式"""
        self.selected_algo = index

        # 重置所有按钮样式
        self.fast_btn.setStyleSheet("""
            QPushButton {
                background-color: #94a3b8;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #64748b;
            }
        """)

        self.fine_btn.setStyleSheet("""
            QPushButton {
                background-color: #94a3b8;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #64748b;
            }
        """)

        self.deep_btn.setStyleSheet("""
            QPushButton {
                background-color: #94a3b8;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #64748b;
            }
        """)

        # 设置选中按钮样式
        if index == 0:
            self.fast_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3b82f6;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                }
            """)
        elif index == 1:
            self.fine_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3b82f6;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                }
            """)
        elif index == 2:
            self.deep_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3b82f6;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 3px 8px;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                }
            """)

    def select_images(self):
        """选择单张或多张图像"""
        if self.single_radio.isChecked():
            # 单张处理
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择需要修复的图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
            )

            if file_path:
                self.image_paths = [file_path]
                self.load_selected_images()

        else:
            # 批量处理
            dir_path = QFileDialog.getExistingDirectory(
                self, "选择图像文件夹", ""
            )

            if dir_path:
                # 获取文件夹中所有图像文件
                image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
                self.image_paths = [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if os.path.splitext(f)[1].lower() in image_extensions
                ]

                if self.image_paths:
                    self.load_selected_images()
                else:
                    QMessageBox.warning(self, "警告", "所选文件夹中未找到图像文件")
                    self.path_label.setText("未选择任何文件")

    def load_selected_images(self):
        """加载选中的图像"""
        self.original_images = []
        self.restored_images = []
        self.current_index = 0

        try:
            # 显示选择信息
            if len(self.image_paths) == 1:
                self.path_label.setText(f"已选择: {os.path.basename(self.image_paths[0])}")
                self.nav_widget.setVisible(False)
            else:
                self.path_label.setText(f"已选择 {len(self.image_paths)} 张图像")
                self.nav_widget.setVisible(True)
                self.update_navigation()

            # 加载图像
            for path in self.image_paths:
                image = cv2.imread(path)
                if image is None:
                    self.log_message(f"无法读取图像: {os.path.basename(path)}")
                    continue

                # 转换为RGB格式
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.original_images.append(rgb_image)
                self.restored_images.append(None)  # 初始化为None

            # 显示第一张图像
            if self.original_images:
                self.display_image(self.original_preview, self.original_images[0])
                self.restored_preview.setText("尚未修复")
                self.restored_preview.setPixmap(QPixmap())

                # 更新按钮状态
                self.process_btn.setEnabled(True if self.output_dir else False)
                self.save_single_btn.setEnabled(False)
                self.save_batch_btn.setEnabled(False)
                
                # 启用标注功能按钮
                self.upload_annotation_btn.setEnabled(True)

                self.log_message(f"已加载 {len(self.original_images)} 张图像")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
            self.log_message(f"加载图像失败: {str(e)}")

    def update_navigation(self):
        """更新导航按钮状态"""
        if len(self.original_images) <= 1:
            self.nav_widget.setVisible(False)
            return

        self.nav_widget.setVisible(True)
        self.image_index_label.setText(f"{self.current_index + 1}/{len(self.original_images)}")
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.original_images) - 1)

    def show_prev_image(self):
        """显示上一张图像"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_navigation()
            self.display_image(self.original_preview, self.original_images[self.current_index])

            # 显示对应的修复图像
            if self.restored_images[self.current_index] is not None:
                self.display_image(self.restored_preview, self.restored_images[self.current_index])
            else:
                self.restored_preview.setText("尚未修复")
                self.restored_preview.setPixmap(QPixmap())

    def show_next_image(self):
        """显示下一张图像"""
        if self.current_index < len(self.original_images) - 1:
            self.current_index += 1
            self.update_navigation()
            self.display_image(self.original_preview, self.original_images[self.current_index])

            # 显示对应的修复图像
            if self.restored_images[self.current_index] is not None:
                self.display_image(self.restored_preview, self.restored_images[self.current_index])
            else:
                self.restored_preview.setText("尚未修复")
                self.restored_preview.setPixmap(QPixmap())

    def display_image(self, label, image_array):
        """在QLabel中显示图像"""
        if image_array is None:
            label.setText("无图像")
            return

        height, width, channels = image_array.shape
        bytes_per_line = channels * width

        # 转换为QImage
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # 缩放以适应显示区域
        scaled_image = q_image.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        label.setPixmap(QPixmap.fromImage(scaled_image))
        label.setText("")  # 清除文本

    def select_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择保存目录", ""
        )

        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
            self.process_btn.setEnabled(True if self.image_paths else False)

    def start_processing(self):
        """开始修复图像"""
        if not self.image_paths or not self.original_images:
            QMessageBox.warning(self, "警告", "请先选择需要修复的图像")
            return

        if not self.output_dir:
            QMessageBox.warning(self, "警告", "请先选择输出目录")
            return

        # 检查是否有可用的掩码文件（可选）
        mask_files = glob.glob(os.path.join(self.temp_mask_dir, "mask_*.png"))
        if mask_files:
            self.log_message(f"检测到 {len(mask_files)} 个掩码文件，将在修复过程中使用")

        # 禁用相关按钮
        self.process_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_single_btn.setEnabled(False)
        self.save_batch_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.fast_btn.setEnabled(False)
        self.fine_btn.setEnabled(False)
        self.deep_btn.setEnabled(False)
        
        # 禁用标注功能
        self.upload_annotation_btn.setEnabled(False)
        self.start_annotation_btn.setEnabled(False)
        self.confirm_annotation_btn.setEnabled(False)
        self.reset_annotation_btn.setEnabled(False)

        # 重置进度条
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在准备修复...")
        self.status_label.setText("正在准备修复...")

        # 创建并启动处理线程
        self.processing_thread = RestorationThread(
            self.original_images,
            self.strength_slider.value(),
            self.selected_algo,
            self.denoise_slider.value(),
            self.detail_slider.value()
        )

        # 连接信号
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.log_updated.connect(self.log_message)
        self.processing_thread.finished.connect(self.processing_finished)

        # 开始处理
        self.is_processing = True
        self.processing_thread.start()

    def update_progress(self, value, text):
        """更新进度条和进度文本"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)

    def update_status(self, text):
        """更新状态标签"""
        self.status_label.setText(text)

    def cancel_processing(self):
        """取消修复处理"""
        if self.processing_thread and self.is_processing:
            self.processing_thread.cancel()
            self.status_label.setText("正在取消修复...")
            self.progress_label.setText("正在取消...")
            self.log_message("用户取消了修复操作")

    def processing_finished(self, result):
        """处理完成回调"""
        self.is_processing = False
        self.cancel_btn.setEnabled(False)
        self.select_btn.setEnabled(True)
        self.fast_btn.setEnabled(True)
        self.fine_btn.setEnabled(True)
        self.deep_btn.setEnabled(True)
        self.prev_btn.setEnabled(len(self.original_images) > 1)
        self.next_btn.setEnabled(len(self.original_images) > 1)

        # 重新启用标注功能
        if self.original_images:
            self.upload_annotation_btn.setEnabled(True)
        self.start_annotation_btn.setEnabled(False)
        self.confirm_annotation_btn.setEnabled(False)
        self.reset_annotation_btn.setEnabled(False)

        # 清理临时掩码文件
        self.cleanupTempMask()

        if result is not None:
            # 处理成功
            self.restored_images = result[0]
            stats = result[1]

            # 显示当前图像的修复结果
            if self.restored_images[self.current_index] is not None:
                self.display_image(self.restored_preview, self.restored_images[self.current_index])

            # 更新状态
            self.status_label.setText("修复完成")
            self.progress_label.setText(f"修复完成 ({stats['count']}/{len(self.original_images)})")
            self.save_single_btn.setEnabled(True)
            self.save_batch_btn.setEnabled(len(self.restored_images) > 0)

            # 更新统计信息
            self.stats_label.setText(f"""
                修复耗时: {stats['time']}
                修复图像: {stats['count']}/{len(self.original_images)}
                平均区域: {stats['avg_area']} 像素
            """)

            # 自动保存
            if len(self.original_images) > 1:
                self.auto_save_restored_images()
            else:
                self.auto_save_current_restored_image()
        else:
            # 处理取消或失败
            self.status_label.setText("修复已取消或失败")
            self.progress_label.setText("修复已取消或失败")
            self.save_single_btn.setEnabled(False)
            self.save_batch_btn.setEnabled(False)

    def on_progress_update(self, progress, text):
        """更新进度条和进度文本"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(text)

    def on_status_update(self, status):
        """更新状态标签"""
        self.status_label.setText(status)

    def auto_save_current_restored_image(self):
        """自动保存当前修复结果（单张）"""
        if not self.restored_images or self.restored_images[self.current_index] is None:
            return

        try:
            # 构建输出路径
            filename = os.path.splitext(os.path.basename(self.image_paths[self.current_index]))[0]
            output_format = self.format_combo.currentText().lower()
            output_path = os.path.join(
                self.output_dir,
                f"{filename}_restored.{output_format}"
            )

            # 转换为BGR格式保存
            bgr_image = cv2.cvtColor(self.restored_images[self.current_index], cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, bgr_image)

            self.log_message(f"自动保存: {os.path.basename(output_path)}")

        except Exception as e:
            self.log_message(f"自动保存失败: {str(e)}")

    def auto_save_restored_images(self):
        """自动批量保存修复结果"""
        if not self.restored_images:
            return

        success_count = 0
        for i, restored_img in enumerate(self.restored_images):
            if restored_img is None:
                continue

            try:
                # 构建输出路径
                filename = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                output_format = self.format_combo.currentText().lower()
                output_path = os.path.join(
                    self.output_dir,
                    f"{filename}_restored.{output_format}"
                )

                # 转换为BGR格式保存
                bgr_image = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, bgr_image)

                success_count += 1
                self.log_message(f"自动保存 [{i + 1}/{len(self.restored_images)}]: {os.path.basename(output_path)}")

            except Exception as e:
                self.log_message(f"保存失败 [{i + 1}/{len(self.restored_images)}]: {str(e)}")

        self.log_message(f"批量保存完成，成功 {success_count}/{len(self.restored_images)} 张")

    def save_current_restored_image(self):
        """手动保存当前修复结果"""
        if not self.restored_images or self.restored_images[self.current_index] is None:
            return

        try:
            # 获取保存路径
            filename = os.path.splitext(os.path.basename(self.image_paths[self.current_index]))[0] + "_restored"
            output_format = self.format_combo.currentText().lower()

            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存修复结果",
                os.path.join(self.output_dir, filename),
                f"{output_format.upper()}文件 (*.{output_format})"
            )

            if file_path:
                # 转换为BGR格式保存
                bgr_image = cv2.cvtColor(self.restored_images[self.current_index], cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr_image)

                self.log_message(f"已保存: {os.path.basename(file_path)}")
                QMessageBox.information(self, "保存成功", "修复结果已成功保存")

        except Exception as e:
            self.log_message(f"保存失败: {str(e)}")
            QMessageBox.critical(self, "保存错误", f"保存失败: {str(e)}")

    def batch_save_restored_images(self):
        """批量保存所有修复结果"""
        if not self.restored_images:
            return

        # 选择保存目录
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择批量保存目录", self.output_dir
        )

        if not dir_path:
            return

        success_count = 0
        for i, restored_img in enumerate(self.restored_images):
            if restored_img is None:
                continue

            try:
                # 构建输出路径
                filename = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                output_format = self.format_combo.currentText().lower()
                output_path = os.path.join(
                    dir_path,
                    f"{filename}_restored.{output_format}"
                )

                # 转换为BGR格式保存
                bgr_image = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, bgr_image)

                success_count += 1
                self.log_message(f"批量保存 [{i + 1}/{len(self.restored_images)}]: {os.path.basename(output_path)}")

            except Exception as e:
                self.log_message(f"批量保存失败 [{i + 1}/{len(self.restored_images)}]: {str(e)}")

        self.log_message(f"批量保存完成，成功 {success_count}/{len(self.restored_images)} 张")
        QMessageBox.information(self, "批量保存完成", f"已成功保存 {success_count} 张修复结果")

    def log_message(self, message):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_list.addItem(f"[{timestamp}] {message}")
        self.log_list.scrollToBottom()  # 滚动到最新条目

    # ==================== 手动污渍标注功能 ====================
    
    def uploadAnnotationImage(self):
        """导入图像用于标注"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择要标注的图像", "", 
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            
            if file_path:
                # 读取图像
                image = cv2.imread(file_path)
                if image is None:
                    QMessageBox.warning(self, "错误", "无法读取图像文件！")
                    return
                
                self.annotation_image = image.copy()
                self.annotation_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                
                # 显示图像
                self.displayAnnotationImage()
                
                # 更新状态
                self.annotation_status_label.setText("状态: 图像已导入，可以开始标注")
                self.annotation_status_label.setStyleSheet("color: #3b82f6; font-weight: bold;")
                
                # 启用按钮
                self.start_annotation_btn.setEnabled(True)
                self.reset_annotation_btn.setEnabled(True)
                
                self.log_message(f"已导入图像用于标注: {os.path.basename(file_path)}")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入图像时发生错误:\n{str(e)}")
    
    def startAnnotation(self):
        """开始标注"""
        if self.annotation_image is None:
            QMessageBox.warning(self, "警告", "请先导入图像！")
            return
        
        self.annotation_enabled = True
        self.annotation_status_label.setText("状态: 标注中...请在图像上涂抹污渍区域")
        self.annotation_status_label.setStyleSheet("color: #10b981; font-weight: bold;")
        
        # 启用/禁用按钮
        self.start_annotation_btn.setEnabled(False)
        self.confirm_annotation_btn.setEnabled(True)
        self.upload_annotation_btn.setEnabled(False)
        
        # 启用鼠标事件
        self.annotation_display.mousePressEvent = self.annotation_mousePressEvent
        self.annotation_display.mouseMoveEvent = self.annotation_mouseMoveEvent
        self.annotation_display.mouseReleaseEvent = self.annotation_mouseReleaseEvent
        
        self.log_message("开始手动污渍标注")
    
    def confirmAnnotation(self):
        """确认标注并生成二值化掩码"""
        if self.annotation_image is None:
            QMessageBox.warning(self, "警告", "请先导入图像！")
            return
        
        try:
            # 生成二值化掩码
            binary_mask = self.createBinaryMask()
            
            # 保存掩码到临时目录
            timestamp = int(time.time())
            mask_filename = f"mask_{timestamp}.png"
            mask_path = os.path.join(self.temp_mask_dir, mask_filename)
            
            cv2.imwrite(mask_path, binary_mask)
            
            # 显示掩码
            self.displayMask(binary_mask)
            
            # 更新状态
            self.annotation_status_label.setText(f"状态: 掩码已保存 ({mask_filename})")
            self.annotation_status_label.setStyleSheet("color: #f59e0b; font-weight: bold;")
            
            # 禁用标注
            self.annotation_enabled = False
            
            # 启用/禁用按钮
            self.confirm_annotation_btn.setEnabled(False)
            self.start_annotation_btn.setEnabled(True)
            self.upload_annotation_btn.setEnabled(True)
            
            self.log_message(f"标注完成，掩码已保存: {mask_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"确认标注时发生错误:\n{str(e)}")
    
    def resetAnnotation(self):
        """重置标注"""
        if self.annotation_image is None:
            return
        
        # 重置掩码
        self.annotation_mask = np.zeros((self.annotation_image.shape[0], self.annotation_image.shape[1]), dtype=np.uint8)
        
        # 重新显示原图
        self.displayAnnotationImage()
        
        # 更新状态
        self.annotation_status_label.setText("状态: 标注已重置")
        self.annotation_status_label.setStyleSheet("color: #94a3b8; font-weight: bold;")
        
        # 重置按钮状态
        self.annotation_enabled = False
        self.start_annotation_btn.setEnabled(True)
        self.confirm_annotation_btn.setEnabled(False)
        
        self.log_message("标注已重置")
    
    def createBinaryMask(self):
        """创建二值化掩码"""
        if self.annotation_mask is None:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # 将掩码二值化（非零像素设为255）
        binary_mask = np.where(self.annotation_mask > 0, 255, 0).astype(np.uint8)
        
        return binary_mask
    
    def displayAnnotationImage(self):
        """显示标注图像"""
        if self.annotation_image is None:
            return
        
        # 将OpenCV图像转换为QPixmap
        rgb_image = cv2.cvtColor(self.annotation_image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 缩放以适应显示区域
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.annotation_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.annotation_display.setPixmap(scaled_pixmap)
    
    def displayMask(self, mask):
        """显示掩码"""
        # 将掩码转换为QPixmap
        height, width = mask.shape
        q_image = QImage(mask.data, width, height, width, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        
        # 缩放以适应显示区域
        scaled_pixmap = pixmap.scaled(
            self.annotation_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.annotation_display.setPixmap(scaled_pixmap)
    
    def updateBrushSize(self, value):
        """更新画笔大小"""
        self.brush_size_value.setText(str(value))
    
    def annotation_mousePressEvent(self, event):
        """标注鼠标按下事件"""
        if not self.annotation_enabled or self.annotation_image is None:
            return
        
        if event.button() == Qt.MouseButton.LeftButton:
            self.painting = True
            self.paintAnnotation(event.position())
    
    def annotation_mouseMoveEvent(self, event):
        """标注鼠标移动事件"""
        if not self.annotation_enabled or not self.painting:
            return
        
        self.paintAnnotation(event.position())
    
    def annotation_mouseReleaseEvent(self, event):
        """标注鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.painting = False
    
    def paintAnnotation(self, pos):
        """在图像上绘制标注"""
        if self.annotation_image is None or self.annotation_mask is None:
            return
        
        # 获取画笔大小
        brush_size = self.brush_size_slider.value()
        
        # 获取显示区域的缩放比例
        pixmap = self.annotation_display.pixmap()
        if pixmap is None:
            return
        
        # 计算图像在显示区域中的实际位置和大小
        scaled_pixmap = pixmap.scaled(
            self.annotation_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        img_width = scaled_pixmap.width()
        img_height = scaled_pixmap.height()
        
        # 计算偏移量（居中显示）
        offset_x = (self.annotation_display.width() - img_width) // 2
        offset_y = (self.annotation_display.height() - img_height) // 2
        
        # 检查点击位置是否在图像范围内
        if (pos.x() < offset_x or pos.x() >= offset_x + img_width or
            pos.y() < offset_y or pos.y() >= offset_y + img_height):
            return
        
        # 映射到原图坐标
        img_x = int((pos.x() - offset_x) * self.annotation_image.shape[1] / img_width)
        img_y = int((pos.y() - offset_y) * self.annotation_image.shape[0] / img_height)
        
        # 确保坐标在图像范围内
        img_x = max(0, min(img_x, self.annotation_image.shape[1] - 1))
        img_y = max(0, min(img_y, self.annotation_image.shape[0] - 1))
        
        # 在掩码上绘制圆形
        cv2.circle(self.annotation_mask, (img_x, img_y), brush_size // 2, 255, -1)
        
        # 更新显示（在原图上显示涂抹效果）
        display_image = self.annotation_image.copy()
        display_image[self.annotation_mask > 0] = [0, 255, 0]  # 用绿色显示涂抹区域
        
        self.displayAnnotationImageWithMask(display_image)
    
    def displayAnnotationImageWithMask(self, image):
        """显示带掩码的标注图像"""
        # 将OpenCV图像转换为QPixmap
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # 缩放以适应显示区域
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.annotation_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.annotation_display.setPixmap(scaled_pixmap)
    
    def loadLatestMask(self):
        """加载最新的掩码文件"""
        try:
            # 查找临时目录中最新的掩码文件
            mask_files = glob.glob(os.path.join(self.temp_mask_dir, "mask_*.png"))
            
            if not mask_files:
                return None
            
            # 按修改时间排序，获取最新的
            latest_mask_file = max(mask_files, key=os.path.getctime)
            
            # 读取掩码
            mask = cv2.imread(latest_mask_file, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                self.log_message(f"已加载最新掩码: {os.path.basename(latest_mask_file)}")
                return mask
            else:
                self.log_message("无法读取掩码文件")
                return None
                
        except Exception as e:
            self.log_message(f"加载掩码时发生错误: {str(e)}")
            return None
    
    def cleanupTempMask(self):
        """清理临时掩码文件"""
        try:
            # 查找临时目录中的掩码文件
            mask_files = glob.glob(os.path.join(self.temp_mask_dir, "mask_*.png"))
            
            for mask_file in mask_files:
                try:
                    os.remove(mask_file)
                    self.log_message(f"已删除临时掩码: {os.path.basename(mask_file)}")
                except Exception as e:
                    self.log_message(f"删除掩码文件失败: {str(e)}")
                    
        except Exception as e:
            self.log_message(f"清理临时掩码时发生错误: {str(e)}")
        """快速修复算法"""
        try:
            # 基础图像增强
            enhanced = cv2.convertScaleAbs(image, alpha=self.strength/50.0, beta=10)
            
            # 如果有掩码，使用inpainting进行局部修复
            if mask is not None:
                # 调整掩码尺寸以匹配图像
                if mask.shape[:2] != image.shape[:2]:
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                else:
                    mask_resized = mask
                
                # 二值化掩码
                mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
                
                # 使用inpainting修复
                restored = cv2.inpaint(enhanced, mask_binary, 3, cv2.INPAINT_TELEA)
                self.log_updated.emit("使用掩码进行局部修复")
            else:
                restored = enhanced
            
            # 降噪处理
            if self.denoise_level > 0:
                restored = cv2.bilateralFilter(restored, 9, self.denoise_level*50, self.denoise_level*50)
            
            return restored
        except Exception as e:
            self.log_updated.emit(f"快速修复算法出错: {str(e)}")
            return None

    def fine_restoration(self, image, mask=None):
        """精细修复算法"""
        try:
            # 边缘保持滤波
            filtered = cv2.edgePreservingFilter(image, flags=1, sigma_s=self.strength/10.0, sigma_r=0.4)
            
            # 如果有掩码，使用inpainting进行局部修复
            if mask is not None:
                # 调整掩码尺寸以匹配图像
                if mask.shape[:2] != image.shape[:2]:
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                else:
                    mask_resized = mask
                
                # 二值化掩码
                mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
                
                # 使用inpainting修复
                restored = cv2.inpaint(filtered, mask_binary, 5, cv2.INPAINT_NS)
                self.log_updated.emit("使用掩码进行精细局部修复")
            else:
                restored = filtered
            
            # 细节增强
            detail_enhancement = self.detail_level / 100.0
            if detail_enhancement > 0:
                # 创建细节层
                detail = cv2.subtract(restored, cv2.GaussianBlur(restored, (0, 0), 3))
                restored = cv2.addWeighted(restored, 1, detail, detail_enhancement, 0)
            
            return restored
        except Exception as e:
            self.log_updated.emit(f"精细修复算法出错: {str(e)}")
            return None

    def deep_restoration(self, image, mask=None):
        """深度修复算法"""
        try:
            # 非局部均值去噪
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None,
                self.denoise_level / 10,
                self.denoise_level / 10,
                7, 21
            )
            
            # 如果有掩码，使用inpainting进行局部修复
            if mask is not None:
                # 调整掩码尺寸以匹配图像
                if mask.shape[:2] != image.shape[:2]:
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                else:
                    mask_resized = mask
                
                # 二值化掩码
                mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
                
                # 使用inpainting修复
                restored = cv2.inpaint(denoised, mask_binary, 7, cv2.INPAINT_NS)
                self.log_updated.emit("使用掩码进行深度局部修复")
            else:
                restored = denoised
            
            return restored
        except Exception as e:
            self.log_updated.emit(f"深度修复算法出错: {str(e)}")
            return None

    def apply_restoration(self, image, mask=None):
        """应用修复算法"""
        # 模拟修复过程
        result = image.copy()

        # 根据选择的算法应用不同的处理
        if self.algorithm == 0:  # 快速修复
            result = self.quick_restoration(image, mask)

        elif self.algorithm == 1:  # 精细修复
            result = self.fine_restoration(image, mask)

        elif self.algorithm == 2:  # 深度修复（模拟）
            result = self.deep_restoration(image, mask)

        return result

    def cancel(self):
        """取消处理"""
        self.canceled = True

class RestorationThread(QThread):
    """图像修复线程，避免界面卡顿"""
    progress_updated = pyqtSignal(int, str)  # 进度值, 进度文本
    status_updated = pyqtSignal(str)  # 状态文本
    log_updated = pyqtSignal(str)  # 日志消息
    finished = pyqtSignal(object)  # 处理结果

    def __init__(self, images, strength, algorithm, denoise_level, detail_level):
        super().__init__()
        self.original_images = images
        self.strength = strength
        self.algorithm = algorithm  # 0:快速, 1:精细, 2:深度
        self.denoise_level = denoise_level
        self.detail_level = detail_level
        self.canceled = False
        self.temp_mask_dir = "D:/python_chuangxin/jiemianpythonProject/V2/zanshibaocun"

    def run(self):
        """线程运行函数"""
        start_time = time.time()
        stats = {
            'time': '00:00:00',
            'count': 0,
            'avg_area': 0
        }

        restored_images = []
        total_area = 0

        try:
            total = len(self.original_images)
            algo_name = ["快速修复", "精细修复", "深度修复"][self.algorithm]
            self.log_updated.emit(f"开始批量修复，使用{algo_name}，共 {total} 张图像")

            # 加载掩码文件（如果有）
            mask = self.loadLatestMask()
            if mask is not None:
                self.log_updated.emit("已加载掩码文件，将在修复过程中使用")

            for i, image_path in enumerate(self.original_images):
                if self.canceled:
                    self.finished.emit(None)
                    return

                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    self.log_updated.emit(f"无法读取图像: {image_path}")
                    continue

                # 更新进度和状态
                progress = int((i + 1) / total * 100)
                progress_text = f"正在处理 {i + 1}/{total} ({progress}%)"
                self.progress_updated.emit(progress, progress_text)
                self.status_updated.emit(f"正在修复 {i + 1}/{total}")
                self.log_updated.emit(f"开始修复第 {i + 1} 张图像")

                # 模拟修复过程
                try:
                    # 模拟检测到的损失区域
                    height, width = image.shape[:2]
                    damage_area = int((height * width) * (0.1 + self.strength * 0.02))
                    total_area += damage_area

                    # 应用修复算法
                    restored_img = self.apply_restoration(image, mask)
                    restored_images.append(restored_img)
                    stats['count'] += 1

                    self.log_updated.emit(f"完成修复第 {i + 1} 张图像")

                except Exception as e:
                    self.log_updated.emit(f"修复第 {i + 1} 张图像失败: {str(e)}")
                    restored_images.append(None)

            if self.canceled:
                self.finished.emit(None)
                return

            # 计算统计信息
            end_time = time.time()
            elapsed = int(end_time - start_time)
            stats['time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            stats['avg_area'] = int(total_area / stats['count']) if stats['count'] > 0 else 0

            self.log_updated.emit(f"全部修复完成，共处理 {stats['count']}/{total} 张图像")
            self.progress_updated.emit(100, f"修复完成 ({stats['count']}/{total})")

            # 返回修复结果
            self.finished.emit(restored_images)

        except Exception as e:
            self.log_updated.emit(f"修复过程出错: {str(e)}")
            self.finished.emit(None)

    def quick_restoration(self, image, mask=None):
        """快速修复算法"""
        try:
            # 基础图像增强
            enhanced = cv2.convertScaleAbs(image, alpha=self.strength/50.0, beta=10)
            
            # 如果有掩码，使用inpainting进行局部修复
            if mask is not None:
                # 调整掩码尺寸以匹配图像
                if mask.shape[:2] != image.shape[:2]:
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                else:
                    mask_resized = mask
                
                # 二值化掩码
                mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
                
                # 使用inpainting修复
                restored = cv2.inpaint(enhanced, mask_binary, 3, cv2.INPAINT_TELEA)
                self.log_updated.emit("使用掩码进行局部修复")
            else:
                restored = enhanced
            
            # 降噪处理
            if self.denoise_level > 0:
                restored = cv2.bilateralFilter(restored, 9, self.denoise_level*50, self.denoise_level*50)
            
            return restored
        except Exception as e:
            self.log_updated.emit(f"快速修复算法出错: {str(e)}")
            return None

    def fine_restoration(self, image, mask=None):
        """精细修复算法"""
        try:
            # 边缘保持滤波
            filtered = cv2.edgePreservingFilter(image, flags=1, sigma_s=self.strength/10.0, sigma_r=0.4)
            
            # 如果有掩码，使用inpainting进行局部修复
            if mask is not None:
                # 调整掩码尺寸以匹配图像
                if mask.shape[:2] != image.shape[:2]:
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                else:
                    mask_resized = mask
                
                # 二值化掩码
                mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
                
                # 使用inpainting修复
                restored = cv2.inpaint(filtered, mask_binary, 5, cv2.INPAINT_NS)
                self.log_updated.emit("使用掩码进行精细局部修复")
            else:
                restored = filtered
            
            # 细节增强
            detail_enhancement = self.detail_level / 100.0
            if detail_enhancement > 0:
                # 创建细节层
                detail = cv2.subtract(restored, cv2.GaussianBlur(restored, (0, 0), 3))
                restored = cv2.addWeighted(restored, 1, detail, detail_enhancement, 0)
            
            return restored
        except Exception as e:
            self.log_updated.emit(f"精细修复算法出错: {str(e)}")
            return None

    def deep_restoration(self, image, mask=None):
        """深度修复算法"""
        try:
            # 非局部均值去噪
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None,
                self.denoise_level / 10,
                self.denoise_level / 10,
                7, 21
            )
            
            # 如果有掩码，使用inpainting进行局部修复
            if mask is not None:
                # 调整掩码尺寸以匹配图像
                if mask.shape[:2] != image.shape[:2]:
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                else:
                    mask_resized = mask
                
                # 二值化掩码
                mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)[1]
                
                # 使用inpainting修复
                restored = cv2.inpaint(denoised, mask_binary, 7, cv2.INPAINT_NS)
                self.log_updated.emit("使用掩码进行深度局部修复")
            else:
                restored = denoised
            
            return restored
        except Exception as e:
            self.log_updated.emit(f"深度修复算法出错: {str(e)}")
            return None

    def apply_restoration(self, image, mask=None):
        """应用修复算法"""
        # 根据选择的算法应用不同的处理
        if self.algorithm == 0:  # 快速修复
            result = self.quick_restoration(image, mask)
        elif self.algorithm == 1:  # 精细修复
            result = self.fine_restoration(image, mask)
        elif self.algorithm == 2:  # 深度修复
            result = self.deep_restoration(image, mask)
        else:
            result = image.copy()
        
        return result

    def cancel(self):
        """取消处理"""
        self.canceled = True

    def loadLatestMask(self):
        """加载最新的掩码文件"""
        try:
            mask_files = glob.glob(os.path.join(self.temp_mask_dir, "mask_*.png"))
            if mask_files:
                # 按修改时间排序，获取最新的掩码文件
                latest_mask_file = max(mask_files, key=os.path.getmtime)
                mask = cv2.imread(latest_mask_file, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    self.log_updated.emit(f"成功加载掩码文件: {os.path.basename(latest_mask_file)}")
                    return mask
                else:
                    self.log_updated.emit(f"无法读取掩码文件: {latest_mask_file}")
            else:
                self.log_updated.emit("未找到掩码文件")
        except Exception as e:
            self.log_updated.emit(f"加载掩码文件时出错: {str(e)}")
        return None


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    # 确保中文显示正常
    font = QFont("SimHei")
    app.setFont(font)

    window = RelicRestorationPage()
    window.show()

    sys.exit(app.exec())
