#!/usr/bin/env python3
import sys
import os
import time
import json
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QProgressBar, QListWidget, QGroupBox,
                             QRadioButton, QComboBox, QMessageBox, QFrame, QCheckBox,
                             QListWidgetItem, QApplication, QToolTip)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPixmap, QImage

# 导入canshu.py中的参数调节模块
try:
    from canshu import ImageParameterAdjuster, integrate_parameter_adjuster, update_preview_with_parameters

    PARAM_ADJUST_AVAILABLE = True
except ImportError as e:
    print(f"canshu.py导入失败: {e}")
    print("参数调节功能将无法使用")
    PARAM_ADJUST_AVAILABLE = False

# 导入predict.py中的快速修复模型
try:
    from predict import FastRestorationModel

    PREDICT_AVAILABLE = True
except ImportError as e:
    print(f"predict.py导入失败: {e}")
    print("快速修复将使用模拟模式")
    PREDICT_AVAILABLE = False


# 修复工作线程（支持参数调节配置传递）
class RestorationWorker(QObject):
    progress_updated = pyqtSignal(int, str)  # 进度值, 进度文本
    status_updated = pyqtSignal(str)  # 状态文本
    log_updated = pyqtSignal(str)  # 日志信息
    finished = pyqtSignal(object)  # 完成信号(结果, 统计)

    def __init__(self, image_paths, mask_path, output_dir, format_type, model_path, device='cpu',
                 adjust_params=None):
        super().__init__()
        self.image_paths = image_paths
        self.mask_path = mask_path
        self.output_dir = output_dir
        self.format_type = format_type
        self.model_path = model_path
        self.device = device
        self.is_cancelled = False
        self.adjust_params = adjust_params if adjust_params else {
            'contrast': False, 'brightness': False, 'color': False
        }

    def cancel(self):
        self.is_cancelled = True
        self.log_updated.emit("收到取消请求，终止任务...")

    def run(self):
        try:
            results = []
            total = len(self.image_paths)
            start_time = time.time()

            self.status_updated.emit("开始快速修复")
            adjust_log = []
            if self.adjust_params['contrast']:
                adjust_log.append("对比度增强")
            if self.adjust_params['brightness']:
                adjust_log.append("亮度调整")
            if self.adjust_params['color']:
                adjust_log.append("色彩平衡")
            self.log_updated.emit(
                f"任务参数: 图像数={total}, 设备={self.device}, 参数调节: {', '.join(adjust_log) if adjust_log else '无'}")

            # 加载LAMA模型
            self.progress_updated.emit(5, "后台加载LAMA模型...")
            try:
                model = FastRestorationModel(
                    model_path=self.model_path,
                    device=self.device,
                    refine=False
                )
                self.log_updated.emit("LAMA模型加载成功")
            except Exception as model_e:
                self.log_updated.emit(f"模型加载失败: {str(model_e)}，切换至模拟修复")
                self._run_simulation(results, total)
                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                stats = {"time": elapsed, "success": len(results), "total": total, "avg_area": "--"}
                self.status_updated.emit("修复完成（模拟）")
                self.finished.emit((results, stats))
                return

            # 初始化参数调整器
            param_adjuster = ImageParameterAdjuster() if PARAM_ADJUST_AVAILABLE else None
            if param_adjuster:
                param_adjuster.set_contrast(1.5)
                param_adjuster.set_brightness(30)
                param_adjuster.set_color_balance(0.9, 1.2, 1.2)
                self.log_updated.emit("参数调整器初始化完成，默认强度：对比度1.5倍、亮度+30、蓝降绿红升")

            # 逐张处理图像
            for idx, img_path in enumerate(self.image_paths):
                if self.is_cancelled:
                    self.status_updated.emit("修复已取消")
                    self.finished.emit(None)
                    return

                progress = 5 + int((idx + 1) / total * 90)
                img_name = os.path.basename(img_path)
                self.progress_updated.emit(progress, f"修复 {idx + 1}/{total}: {img_name}")
                self.log_updated.emit(f"开始处理: {img_name}")

                # 准备输出路径
                output_path = os.path.join(
                    self.output_dir,
                    f"{os.path.splitext(img_name)[0]}_fast_restored.{self.format_type.lower()}"
                )

                # 处理掩码
                used_mask = self.mask_path
                if not used_mask or not os.path.exists(used_mask):
                    used_mask = self._create_default_mask(img_path)
                    self.log_updated.emit("未选择掩码，创建默认全黑掩码（无损伤修复）")

                # 执行修复
                try:
                    result_bgr = model.predict(
                        image_path=img_path,
                        mask_path=used_mask,
                        output_path=None,
                        out_ext=self.format_type.lower()
                    )
                except Exception as predict_e:
                    self.log_updated.emit(f"处理{img_name}失败: {str(predict_e)}")
                    continue

                # 应用参数调节
                if result_bgr is not None and PARAM_ADJUST_AVAILABLE and param_adjuster:
                    self.log_message(f"对{img_name}应用参数调节")
                    result_bgr = param_adjuster.apply_all_adjustments(
                        result_bgr,
                        apply_contrast=self.adjust_params['contrast'],
                        apply_brightness=self.adjust_params['brightness'],
                        apply_color=self.adjust_params['color']
                    )
                    cv.imwrite(output_path, result_bgr)
                elif result_bgr is not None:
                    cv.imwrite(output_path, result_bgr)

                # 转换为RGB格式存储
                if result_bgr is not None:
                    result_rgb = cv.cvtColor(result_bgr, cv.COLOR_BGR2RGB)
                    results.append((result_rgb, output_path))
                    self.log_updated.emit(f"处理成功: {img_name} -> 保存至{os.path.basename(output_path)}")
                else:
                    self.log_updated.emit(f"处理失败: {img_name}（无返回结果）")

            # 整理结果
            self.progress_updated.emit(95, "整理修复结果...")
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            stats = {
                "time": elapsed,
                "success": len(results),
                "total": total,
                "avg_area": "--"
            }

            self.status_updated.emit("快速修复完成")
            self.log_updated.emit(f"任务结束: 耗时{elapsed}, 成功{len(results)}/{total}张")
            self.finished.emit((results, stats))

        except Exception as e:
            self.status_updated.emit("修复失败")
            self.log_updated.emit(f"任务异常: {str(e)}")
            self.finished.emit(None)

    def _run_simulation(self, results, total):
        param_adjuster = ImageParameterAdjuster() if PARAM_ADJUST_AVAILABLE else None
        if param_adjuster:
            param_adjuster.set_contrast(1.5)
            param_adjuster.set_brightness(30)
            param_adjuster.set_color_balance(0.9, 1.2, 1.2)

        for idx, img_path in enumerate(self.image_paths):
            if self.is_cancelled:
                return

            progress = int((idx + 1) / total * 100)
            img_name = os.path.basename(img_path)
            self.progress_updated.emit(progress, f"模拟修复 {idx + 1}/{total}: {img_name}")
            self.log_updated.emit(f"模拟处理: {img_name}")
            time.sleep(0.3)

            src = cv.imread(img_path)
            if src is not None:
                if PARAM_ADJUST_AVAILABLE and param_adjuster:
                    src = param_adjuster.apply_all_adjustments(
                        src,
                        apply_contrast=self.adjust_params['contrast'],
                        apply_brightness=self.adjust_params['brightness'],
                        apply_color=self.adjust_params['color']
                    )
                result_rgb = cv.cvtColor(src, cv.COLOR_BGR2RGB)
                output_path = os.path.join(
                    self.output_dir,
                    f"{os.path.splitext(img_name)[0]}_sim_restored.{self.format_type.lower()}"
                )
                cv.imwrite(output_path, src)
                results.append((result_rgb, output_path))
                self.log_updated.emit(f"模拟完成: {img_name} -> 保存至{os.path.basename(output_path)}")

    def _create_default_mask(self, img_path):
        img = cv.imread(img_path)
        if img is None:
            raise Exception(f"无法读取图像 {os.path.basename(img_path)} 以创建掩码")

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask_dir = os.path.join(self.output_dir, "temp_masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f"default_mask_{os.path.basename(img_path)}")
        cv.imwrite(mask_path, mask)
        return mask_path

    def log_message(self, message):
        self.log_updated.emit(message)


class RelicRestorationPage(QWidget):
    back_to_initial = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.original_images = []
        self.restored_images = []
        self.restored_paths = []
        self.image_paths = []
        self.mask_path = ""
        self.current_index = 0
        self.output_dir = ""
        self.processing_thread = None
        self.worker = None
        self.is_processing = False
        self.selected_algo = 0
        self.recent_files = []
        self.max_recent_files = 5
        self.recent_file_path = "recent_files.json"
        # 写死LAMA模型路径（需根据实际路径修改）
        self.lama_model_path = r"D:\python_chuangxin\jiemianpythonProject\lama\big-lama\models"
        self.device = "cpu"

        # 参数状态
        self.contrast_enhance = False
        self.brightness_adjust = False
        self.color_balance = False

        # 初始化参数调整器
        self.param_adjuster = ImageParameterAdjuster() if PARAM_ADJUST_AVAILABLE else None

        # 加载最近文件
        self.load_recent_files()
        self.init_ui()

        # 集成参数调节模块
        if PARAM_ADJUST_AVAILABLE:
            integrate_parameter_adjuster(self)
            self.log_message("参数调节模块已成功集成，支持实时预览和结果保存")
        else:
            self.log_message("参数调节模块不可用，相关复选框已禁用")
            self.contrast_checkbox.setEnabled(False)
            self.brightness_checkbox.setEnabled(False)
            self.color_balance_checkbox.setEnabled(False)

    def init_ui(self):
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setMinimumWidth(260)
        left_panel.setMaximumWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        # 标题和返回按钮
        top_layout = QHBoxLayout()
        title_label = QLabel("文物图像修复")
        title_label.setFont(QFont("SimHei", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #1a365d;")
        top_layout.addWidget(title_label)
        top_layout.addStretch()

        back_btn = QPushButton("返回首页")
        back_btn.setFont(QFont("SimHei", 9))
        back_btn.setStyleSheet("""
            QPushButton {background-color: #3b82f6; color: white; border: none; border-radius: 4px; padding: 4px 8px;}
            QPushButton:hover {background-color: #2563eb;}
        """)
        back_btn.setMinimumHeight(25)
        back_btn.clicked.connect(self.back_to_initial.emit)
        top_layout.addWidget(back_btn)
        left_layout.addLayout(top_layout)

        # 1. 图像选择区域
        select_group = QGroupBox("图像选择")
        select_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        select_layout = QVBoxLayout(select_group)
        select_layout.setContentsMargins(8, 8, 8, 8)
        select_layout.setSpacing(6)

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
        self.select_btn.setMinimumHeight(26)
        self.select_btn.setStyleSheet("""
            QPushButton {background-color: #e0f2fe; color: #1a365d; border: 1px solid #93c5fd; border-radius: 4px;}
            QPushButton:hover {background-color: #bfdbfe;}
        """)
        self.select_btn.clicked.connect(self.select_images)
        select_layout.addWidget(self.select_btn)

        # 路径显示
        self.path_label = QLabel("未选择任何文件")
        self.path_label.setFont(QFont("SimHei", 8))
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #4a5568; border: 1px solid #e2e8f0; padding: 4px; border-radius: 3px;")
        select_layout.addWidget(self.path_label)

        # 掩码选择
        mask_layout = QHBoxLayout()
        mask_label = QLabel("掩码图像:")
        mask_label.setFont(QFont("SimHei", 9))
        self.mask_path_label = QLabel("未选择")
        self.mask_path_label.setFont(QFont("SimHei", 8))
        self.mask_path_label.setStyleSheet(
            "color: #4a5568; border: 1px solid #e2e8f0; padding: 2px; border-radius: 2px;")
        select_mask_btn = QPushButton("浏览")
        select_mask_btn.setFont(QFont("SimHei", 8))
        select_mask_btn.setStyleSheet("""
            QPushButton {background-color: #e0f2fe; color: #1a365d; border: 1px solid #93c5fd; border-radius: 3px; padding: 2px 6px;}
        """)
        select_mask_btn.clicked.connect(self.select_mask_image)
        mask_layout.addWidget(mask_label)
        mask_layout.addWidget(self.mask_path_label, 1)
        mask_layout.addWidget(select_mask_btn)
        select_layout.addLayout(mask_layout)

        left_layout.addWidget(select_group)

        # 2. 算法选择区域
        algo_group = QGroupBox("算法选择")
        algo_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setContentsMargins(8, 8, 8, 8)
        algo_layout.setSpacing(6)

        # 快速修复
        fast_layout = QHBoxLayout()
        fast_label = QLabel("快速修复：轻度损伤，速度快")
        fast_label.setFont(QFont("SimHei", 9))
        fast_label.setStyleSheet("color: #4a5568;")
        fast_label.setWordWrap(True)
        self.fast_btn = QPushButton("选择")
        self.fast_btn.setFont(QFont("SimHei", 8))
        self.fast_btn.setStyleSheet("""
            QPushButton {background-color: #3b82f6; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
            QPushButton:hover {background-color: #2563eb;}
        """)
        self.fast_btn.clicked.connect(lambda: self.set_selected_algo(0))
        fast_layout.addWidget(fast_label, 1)
        fast_layout.addWidget(self.fast_btn)
        algo_layout.addLayout(fast_layout)

        # 设备选择
        device_layout = QHBoxLayout()
        device_label = QLabel("运行设备:")
        device_label.setFont(QFont("SimHei", 9))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "GPU"])
        self.device_combo.setCurrentText("CPU")
        self.device_combo.setFont(QFont("SimHei", 9))
        self.device_combo.setMinimumHeight(22)
        self.device_combo.currentTextChanged.connect(self.on_device_change)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        algo_layout.addLayout(device_layout)

        # 精细修复
        fine_layout = QHBoxLayout()
        fine_label = QLabel("精细修复：需要掩码，效果好")
        fine_label.setFont(QFont("SimHei", 9))
        fine_label.setStyleSheet("color: #4a5568;")
        fine_label.setWordWrap(True)
        self.fine_btn = QPushButton("选择")
        self.fine_btn.setFont(QFont("SimHei", 8))
        self.fine_btn.setStyleSheet("""
            QPushButton {background-color: #94a3b8; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
            QPushButton:hover {background-color: #64748b;}
        """)
        self.fine_btn.clicked.connect(lambda: self.set_selected_algo(1))
        fine_layout.addWidget(fine_label, 1)
        fine_layout.addWidget(self.fine_btn)
        algo_layout.addLayout(fine_layout)

        left_layout.addWidget(algo_group)

        # 3. 参数调节区域
        param_group = QGroupBox("参数调节")
        param_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        param_layout = QVBoxLayout(param_group)
        param_layout.setContentsMargins(8, 8, 8, 8)
        param_layout.setSpacing(8)

        # 参数说明
        adjust_note = QLabel("注意勾选后直接生效，在431行该参数")
        adjust_note.setFont(QFont("SimHei", 8))
        adjust_note.setStyleSheet("color: #64748b;")
        param_layout.addWidget(adjust_note)

        # 对比度复选框
        self.contrast_checkbox = QCheckBox("对比度增强（默认1.5倍）")
        self.contrast_checkbox.setFont(QFont("SimHei", 9))
        self.contrast_checkbox.setStyleSheet("color: #4a5568;")
        self.contrast_checkbox.stateChanged.connect(self.on_contrast_changed)
        param_layout.addWidget(self.contrast_checkbox)

        # 亮度复选框
        self.brightness_checkbox = QCheckBox("亮度调整（默认+30）")
        self.brightness_checkbox.setFont(QFont("SimHei", 9))
        self.brightness_checkbox.setStyleSheet("color: #4a5568;")
        self.brightness_checkbox.stateChanged.connect(self.on_brightness_changed)
        param_layout.addWidget(self.brightness_checkbox)

        # 色彩平衡复选框
        self.color_balance_checkbox = QCheckBox("色彩平衡（默认蓝降绿红升）")
        self.color_balance_checkbox.setFont(QFont("SimHei", 9))
        self.color_balance_checkbox.setStyleSheet("color: #4a5568;")
        self.color_balance_checkbox.stateChanged.connect(self.on_color_balance_changed)
        param_layout.addWidget(self.color_balance_checkbox)

        left_layout.addWidget(param_group)

        # 4. 最近处理区域
        recent_group = QGroupBox("最近处理")
        recent_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        recent_layout = QVBoxLayout(recent_group)
        recent_layout.setContentsMargins(8, 8, 8, 8)
        recent_layout.setSpacing(6)

        self.recent_list = QListWidget()
        self.recent_list.setFont(QFont("SimHei", 9))
        self.recent_list.setStyleSheet("""
            QListWidget {border: 1px solid #e2e8f0; border-radius: 4px; padding: 2px;}
            QListWidget::item:hover {background-color: #dbeafe; border-radius: 2px;}
            QListWidget::item:selected {background-color: #bfdbfe; color: #1e40af; border-radius: 2px;}
        """)
        self.recent_list.itemClicked.connect(self.on_recent_item_clicked)
        self.recent_list.setToolTip("点击重新加载文件")
        recent_layout.addWidget(self.recent_list)

        clear_recent_btn = QPushButton("清空列表")
        clear_recent_btn.setFont(QFont("SimHei", 8))
        clear_recent_btn.setStyleSheet("""
            QPushButton {background-color: #e0e7ff; color: #4338ca; border: 1px solid #c7d2fe; border-radius: 3px; padding: 2px 6px;}
            QPushButton:hover {background-color: #c7d2fe;}
        """)
        clear_recent_btn.clicked.connect(self.clear_recent_files)
        recent_layout.addWidget(clear_recent_btn)

        left_layout.addWidget(recent_group)

        # 5. 输出设置区域
        output_group = QGroupBox("输出设置")
        output_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.setSpacing(6)

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
            QPushButton {background-color: #e0f2fe; color: #1a365d; border: 1px solid #93c5fd; border-radius: 3px; padding: 2px 6px;}
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

        # 6. 操作按钮区域
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(6)

        # 开始修复按钮
        self.process_btn = QPushButton("开始修复")
        self.process_btn.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        self.process_btn.setMinimumHeight(30)
        self.process_btn.setStyleSheet("""
            QPushButton {background-color: #10b981; color: white; border: none; border-radius: 5px;}
            QPushButton:hover {background-color: #059669;}
            QPushButton:disabled {background-color: #a7f3d0; color: #64748b;}
        """)
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        btn_layout.addWidget(self.process_btn)

        # 取消按钮
        self.cancel_btn = QPushButton("取消修复")
        self.cancel_btn.setFont(QFont("SimHei", 9))
        self.cancel_btn.setMinimumHeight(26)
        self.cancel_btn.setStyleSheet("""
            QPushButton {background-color: #ef4444; color: white; border: none; border-radius: 5px;}
            QPushButton:hover {background-color: #dc2626;}
            QPushButton:disabled {background-color: #fecaca; color: #64748b;}
        """)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)

        # 保存当前按钮
        self.save_single_btn = QPushButton("保存当前结果")
        self.save_single_btn.setFont(QFont("SimHei", 9))
        self.save_single_btn.setMinimumHeight(26)
        self.save_single_btn.setStyleSheet("""
            QPushButton {background-color: #3b82f6; color: white; border: none; border-radius: 5px;}
            QPushButton:hover {background-color: #2563eb;}
            QPushButton:disabled {background-color: #93c5fd; color: #64748b;}
        """)
        self.save_single_btn.clicked.connect(self.save_current_restored_image)
        self.save_single_btn.setEnabled(False)
        btn_layout.addWidget(self.save_single_btn)

        # 批量保存按钮
        self.save_batch_btn = QPushButton("批量保存结果")
        self.save_batch_btn.setFont(QFont("SimHei", 9))
        self.save_batch_btn.setMinimumHeight(26)
        self.save_batch_btn.setStyleSheet("""
            QPushButton {background-color: #8b5cf6; color: white; border: none; border-radius: 5px;}
            QPushButton:hover {background-color: #7c3aed;}
            QPushButton:disabled {background-color: #c4b5fd; color: #64748b;}
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
        center_layout.setSpacing(8)

        # 顶部标题和导航
        top_center_layout = QHBoxLayout()
        preview_label = QLabel("图像修复前后对比")
        preview_label.setFont(QFont("SimHei", 12, QFont.Weight.Bold))
        preview_label.setStyleSheet("color: #1a365d;")
        top_center_layout.addWidget(preview_label)

        # 批量导航控件
        self.nav_widget = QWidget()
        nav_layout = QHBoxLayout(self.nav_widget)
        nav_layout.setSpacing(8)

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.setFont(QFont("SimHei", 9))
        self.prev_btn.setMinimumHeight(26)
        self.prev_btn.setStyleSheet("""
            QPushButton {background-color: #e2e8f0; color: #1e293b; border: none; border-radius: 3px; padding: 3px 10px;}
            QPushButton:hover {background-color: #cbd5e1;}
            QPushButton:disabled {background-color: #f1f5f9; color: #94a3b8;}
        """)
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.prev_btn.setEnabled(False)

        self.image_index_label = QLabel("")
        self.image_index_label.setFont(QFont("SimHei", 9))

        self.next_btn = QPushButton("下一张")
        self.next_btn.setFont(QFont("SimHei", 9))
        self.next_btn.setMinimumHeight(26)
        self.next_btn.setStyleSheet("""
            QPushButton {background-color: #e2e8f0; color: #1e293b; border: none; border-radius: 3px; padding: 3px 10px;}
            QPushButton:hover {background-color: #cbd5e1;}
            QPushButton:disabled {background-color: #f1f5f9; color: #94a3b8;}
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

        # 图像显示区域
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        preview_layout.setSpacing(6)

        # 修复前图像
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
        self.original_preview.setMinimumSize(450, 280)
        self.original_preview.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; color: #64748b;")
        original_layout.addWidget(self.original_preview)
        preview_layout.addWidget(original_container)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        preview_layout.addWidget(line)

        # 修复后图像
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
        self.restored_preview.setMinimumSize(450, 280)
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

        # 右侧面板
        right_panel = QWidget()
        right_panel.setMinimumWidth(280)
        right_panel.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        # 1. 修复进度
        progress_group = QGroupBox("修复进度")
        progress_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(8, 8, 8, 8)
        progress_layout.setSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {border: 1px solid #cbd5e1; border-radius: 4px; text-align: center;}
            QProgressBar::chunk {background-color: #3b82f6; border-radius: 3px;}
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

        # 3. 处理日志
        log_group = QGroupBox("处理日志")
        log_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)

        self.log_list = QListWidget()
        self.log_list.setFont(QFont("SimHei", 9))
        log_layout.addWidget(self.log_list)
        right_layout.addWidget(log_group, 1)

        main_layout.addWidget(right_panel)
        self.setLayout(main_layout)
        self.update_recent_list_display()

    # ------------------------------
    # 基础功能实现
    # ------------------------------
    def load_image_to_preview(self, file_path, preview_label):
        try:
            image = QImage(file_path)
            if image.isNull():
                raise Exception("无法加载图像文件")

            preview_size = preview_label.size()
            scaled_image = image.scaled(
                preview_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            pixmap = QPixmap.fromImage(scaled_image)
            preview_label.setPixmap(pixmap)
            preview_label.setText("")
            return image
        except Exception as e:
            preview_label.setText(f"加载失败: {str(e)}")
            self.log_message(f"图像加载错误: {str(e)}")
            return None

    def select_mask_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择掩码图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if file_path:
            self.mask_path = file_path
            self.mask_path_label.setText(os.path.basename(file_path))
            self.log_message(f"已选择掩码图像: {os.path.basename(file_path)}")

            mask_preview = QLabel()
            mask_preview.setWindowTitle("掩码图像预览")
            mask_img = QImage(file_path)
            if not mask_img.isNull():
                scaled_mask = mask_img.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
                mask_preview.setPixmap(QPixmap.fromImage(scaled_mask))
                mask_preview.show()

    def load_recent_files(self):
        try:
            if os.path.exists(self.recent_file_path):
                with open(self.recent_file_path, 'r', encoding='utf-8') as f:
                    self.recent_files = json.load(f)

                valid_files = [f for f in self.recent_files if os.path.exists(f)]
                self.recent_files = valid_files
                self.save_recent_files()
        except Exception as e:
            self.log_message(f"加载最近文件失败: {e}")
            self.recent_files = []

    def save_recent_files(self):
        try:
            with open(self.recent_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.recent_files, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log_message(f"保存最近文件失败: {e}")

    def add_recent_file(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return

        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]

        self.save_recent_files()
        self.update_recent_list_display()
        self.log_message(f"已添加到最近处理: {os.path.basename(file_path)}")

    def update_recent_list_display(self):
        self.recent_list.clear()
        for file_path in self.recent_files:
            item = QListWidgetItem(os.path.basename(file_path))
            item.setData(Qt.ItemDataRole.ToolTipRole, file_path)
            self.recent_list.addItem(item)

    def clear_recent_files(self):
        if self.recent_files:
            self.recent_files = []
            self.save_recent_files()
            self.update_recent_list_display()
            self.log_message("已清空最近处理文件列表")

    def on_recent_item_clicked(self, item):
        file_path = item.data(Qt.ItemDataRole.ToolTipRole)
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "文件不存在", "所选文件不存在或已被移动")
            if file_path in self.recent_files:
                self.recent_files.remove(file_path)
                self.save_recent_files()
                self.update_recent_list_display()
            return

        if os.path.isfile(file_path):
            self.single_radio.setChecked(True)
            self.image_paths = [file_path]
            self.path_label.setText(f"已选择: {os.path.basename(file_path)}")

            self.original_images = []
            img = self.load_image_to_preview(file_path, self.original_preview)
            if img:
                self.original_images.append(img)

            self.nav_widget.setVisible(False)
            self.process_btn.setEnabled(True if self.output_dir else False)
            self.restored_preview.setText("尚未修复")
            self.log_message(f"从最近列表加载: {os.path.basename(file_path)}")
        elif os.path.isdir(file_path):
            self.batch_radio.setChecked(True)
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
            self.image_paths = [
                os.path.join(file_path, f)
                for f in os.listdir(file_path)
                if os.path.splitext(f)[1].lower() in image_extensions
            ]

            if self.image_paths:
                self.path_label.setText(f"已选择 {len(self.image_paths)} 张图像")
                self.nav_widget.setVisible(True)
                self.current_index = 0

                self.original_images = []
                for path in self.image_paths:
                    img = QImage(path)
                    if not img.isNull():
                        self.original_images.append(img)

                if self.original_images:
                    self.display_current_image()

                self.update_navigation()
                self.process_btn.setEnabled(True if self.output_dir else False)
                self.restored_preview.setText("尚未修复")
                self.log_message(f"从最近列表加载文件夹: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "警告", "所选文件夹中未找到图像文件")
                self.path_label.setText("未选择任何文件")

    def set_selected_algo(self, index):
        self.selected_algo = index

        # 重置按钮样式
        self.fast_btn.setStyleSheet("""
            QPushButton {background-color: #94a3b8; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
            QPushButton:hover {background-color: #64748b;}
        """)
        self.fine_btn.setStyleSheet("""
            QPushButton {background-color: #94a3b8; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
            QPushButton:hover {background-color: #64748b;}
        """)

        # 设置选中样式
        if index == 0:
            self.fast_btn.setStyleSheet("""
                QPushButton {background-color: #3b82f6; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
                QPushButton:hover {background-color: #2563eb;}
            """)
        elif index == 1:
            self.fine_btn.setStyleSheet("""
                QPushButton {background-color: #3b82f6; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
                QPushButton:hover {background-color: #2563eb;}
            """)

    def select_images(self):
        if self.single_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择需要修复的图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
            )
            if file_path:
                self.image_paths = [file_path]
                self.path_label.setText(f"已选择: {os.path.basename(self.image_paths[0])}")
                self.nav_widget.setVisible(False)
                self.process_btn.setEnabled(True if self.output_dir else False)

                self.original_images = []
                img = self.load_image_to_preview(file_path, self.original_preview)
                if img:
                    self.original_images.append(img)

                self.restored_preview.setText("尚未修复")
                self.add_recent_file(file_path)
        else:
            dir_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹", "")
            if dir_path:
                image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
                self.image_paths = [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if os.path.splitext(f)[1].lower() in image_extensions
                ]

                if self.image_paths:
                    self.path_label.setText(f"已选择 {len(self.image_paths)} 张图像")
                    self.nav_widget.setVisible(True)
                    self.current_index = 0

                    self.original_images = []
                    for path in self.image_paths:
                        img = QImage(path)
                        if not img.isNull():
                            self.original_images.append(img)

                    if self.original_images:
                        self.display_current_image()

                    self.update_navigation()
                    self.process_btn.setEnabled(True if self.output_dir else False)
                    self.restored_preview.setText("尚未修复")
                    self.add_recent_file(dir_path)
                else:
                    QMessageBox.warning(self, "警告", "所选文件夹中未找到图像文件")
                    self.path_label.setText("未选择任何文件")

    def display_current_image(self):
        if 0 <= self.current_index < len(self.original_images):
            image = self.original_images[self.current_index]
            preview_size = self.original_preview.size()
            scaled_image = image.scaled(preview_size, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
            pixmap = QPixmap.fromImage(scaled_image)
            self.original_preview.setPixmap(pixmap)
            self.original_preview.setText("")

    def update_navigation(self):
        if len(self.image_paths) <= 1:
            self.nav_widget.setVisible(False)
            return

        self.nav_widget.setVisible(True)
        self.image_index_label.setText(f"{self.current_index + 1}/{len(self.image_paths)}")
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.image_paths) - 1)

    def show_prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_navigation()
            self.display_current_image()
            self.restored_preview.setText("尚未修复")
            if PARAM_ADJUST_AVAILABLE and self.restored_images:
                update_preview_with_parameters(self)

    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_navigation()
            self.display_current_image()
            self.restored_preview.setText("尚未修复")
            if PARAM_ADJUST_AVAILABLE and self.restored_images:
                update_preview_with_parameters(self)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录", "")
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
            self.process_btn.setEnabled(True if self.image_paths else False)

    def save_current_restored_image(self):
        if not self.image_paths or not self.restored_images:
            return

        filename = os.path.splitext(os.path.basename(self.image_paths[self.current_index]))[0] + "_restored"
        output_format = self.format_combo.currentText().lower()
        default_path = os.path.join(self.output_dir, f"{filename}.{output_format}")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存修复结果", default_path, f"{output_format.upper()}文件 (*.{output_format})"
        )
        if file_path:
            if 0 <= self.current_index < len(self.restored_images):
                result_rgb = self.restored_images[self.current_index]
                result_bgr = cv.cvtColor(result_rgb, cv.COLOR_RGB2BGR)

                # 应用参数调节
                if PARAM_ADJUST_AVAILABLE and self.param_adjuster:
                    result_bgr = self.param_adjuster.apply_all_adjustments(
                        result_bgr,
                        apply_contrast=self.contrast_enhance,
                        apply_brightness=self.brightness_adjust,
                        apply_color=self.color_balance
                    )

                cv.imwrite(file_path, result_bgr)
                self.log_message(f"已保存: {os.path.basename(file_path)}（含参数调节效果）")
                QMessageBox.information(self, "保存成功", "修复结果已成功保存（含参数调节效果）")

    def batch_save_restored_images(self):
        if not self.image_paths or not self.restored_images:
            return

        dir_path = QFileDialog.getExistingDirectory(self, "选择批量保存目录", self.output_dir)
        if dir_path:
            success_count = 0
            total_count = len(self.restored_images)
            for i, (result_rgb, img_path) in enumerate(zip(self.restored_images, self.image_paths)):
                try:
                    filename = os.path.splitext(os.path.basename(img_path))[0] + "_restored"
                    output_format = self.format_combo.currentText().lower()
                    file_path = os.path.join(dir_path, f"{filename}.{output_format}")

                    result_bgr = cv.cvtColor(result_rgb, cv.COLOR_RGB2BGR)
                    # 应用参数调节
                    if PARAM_ADJUST_AVAILABLE and self.param_adjuster:
                        result_bgr = self.param_adjuster.apply_all_adjustments(
                            result_bgr,
                            apply_contrast=self.contrast_enhance,
                            apply_brightness=self.brightness_adjust,
                            apply_color=self.color_balance
                        )

                    cv.imwrite(file_path, result_bgr)
                    success_count += 1
                    self.log_message(f"批量保存: {os.path.basename(file_path)}（含参数调节效果）")
                except Exception as e:
                    self.log_message(f"批量保存失败 {os.path.basename(img_path)}: {str(e)}")

            self.log_message(f"批量保存完成，成功 {success_count}/{total_count} 张（含参数调节效果）")
            QMessageBox.information(self, "批量保存完成",
                                    f"已成功保存 {success_count}/{total_count} 张修复结果（含参数调节效果）")

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_list.addItem(f"[{timestamp}] {message}")
        self.log_list.scrollToBottom()

    # ------------------------------
    # 参数调节回调（由canshu.py覆盖）
    # ------------------------------
    def on_contrast_changed(self, state):
        self.contrast_enhance = state == Qt.CheckState.Checked.value
        self.log_message(f"对比度增强: {'开启' if self.contrast_enhance else '关闭'}")

    def on_brightness_changed(self, state):
        self.brightness_adjust = state == Qt.CheckState.Checked.value
        self.log_message(f"亮度调整: {'开启' if self.brightness_adjust else '关闭'}")

    def on_color_balance_changed(self, state):
        self.color_balance = state == Qt.CheckState.Checked.value
        self.log_message(f"色彩平衡: {'开启' if self.color_balance else '关闭'}")

    def closeEvent(self, event):
        self.save_recent_files()
        event.accept()

    # ------------------------------
    # 设备选择处理
    # ------------------------------
    def on_device_change(self, text):
        self.device = "cuda" if text == "GPU" else "cpu"
        self.log_message(f"运行设备已设置为: {text}")

        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    QMessageBox.warning(self, "警告", "未检测到可用GPU，将自动使用CPU")
                    self.device_combo.setCurrentText("CPU")
                    self.device = "cpu"
            except ImportError:
                QMessageBox.warning(self, "警告", "未安装PyTorch，将自动使用CPU")
                self.device_combo.setCurrentText("CPU")
                self.device = "cpu"

    # ------------------------------
    # 进度和状态更新
    # ------------------------------
    def update_progress(self, value, text):
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)

    def update_status(self, text):
        self.status_label.setText(text)

    # ------------------------------
    # 修复完成回调
    # ------------------------------
    def processing_finished(self, result):
        self.is_processing = False

        # 恢复UI
        self.cancel_btn.setEnabled(False)
        self.select_btn.setEnabled(True)
        self.fast_btn.setEnabled(True)
        self.fine_btn.setEnabled(True)
        self.contrast_checkbox.setEnabled(True)
        self.brightness_checkbox.setEnabled(True)
        self.color_balance_checkbox.setEnabled(True)
        self.prev_btn.setEnabled(len(self.image_paths) > 1)
        self.next_btn.setEnabled(len(self.image_paths) > 1)
        self.process_btn.setEnabled(True)
        self.recent_list.setEnabled(True)
        self.device_combo.setEnabled(True)

        if result is not None:
            results, stats = result
            self.restored_images = [res[0] for res in results]
            self.restored_paths = [res[1] for res in results]

            # 应用参数调节并更新预览
            if PARAM_ADJUST_AVAILABLE:
                self.log_message("修复完成，自动应用参数调节并更新预览")
                update_preview_with_parameters(self)

            # 更新统计
            self.stats_label.setText(f"""
                修复耗时: {stats['time']}
                修复图像: {stats['success']}/{stats['total']}
                平均区域: {stats['avg_area']} 像素
            """)

            # 无参数调节时显示结果
            if self.restored_images and 0 <= self.current_index < len(
                    self.restored_images) and not PARAM_ADJUST_AVAILABLE:
                result_rgb = self.restored_images[self.current_index]
                h, w, ch = result_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                scaled_image = qt_image.scaled(self.restored_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
                self.restored_preview.setPixmap(QPixmap.fromImage(scaled_image))
                self.restored_preview.setText("")

            # 更新保存按钮
            self.save_single_btn.setEnabled(len(self.restored_images) > 0)
            self.save_batch_btn.setEnabled(len(self.restored_images) > 1)

            # 添加到最近文件
            if self.image_paths:
                if len(self.image_paths) == 1:
                    self.add_recent_file(self.image_paths[0])
                else:
                    self.add_recent_file(os.path.dirname(self.image_paths[0]))

    # ------------------------------
    # 开始修复
    # ------------------------------
    def start_processing(self):
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请先选择需要修复的图像")
            return

        if not self.output_dir:
            QMessageBox.warning(self, "警告", "请先选择输出目录")
            return

        # 精细修复需掩码
        if self.selected_algo == 1 and (not self.mask_path or not os.path.exists(self.mask_path)):
            QMessageBox.warning(self, "警告", "精细修复需要选择有效的掩码图像")
            return

        # 检查快速修复模型路径
        if self.selected_algo == 0:
            if not os.path.exists(self.lama_model_path):
                QMessageBox.warning(self, "警告", f"LAMA模型路径不存在: {self.lama_model_path}")
                return

            required_files = [
                os.path.join(self.lama_model_path, 'config.yaml'),
                os.path.join(self.lama_model_path, 'best.ckpt')
            ]
            if not all(os.path.exists(f) for f in required_files):
                missing = [os.path.basename(f) for f in required_files if not os.path.exists(f)]
                QMessageBox.warning(self, "警告", f"模型目录不完整，缺少: {', '.join(missing)}")
                return

        # 禁用UI
        self.process_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.save_single_btn.setEnabled(False)
        self.save_batch_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.fast_btn.setEnabled(False)
        self.fine_btn.setEnabled(False)
        self.contrast_checkbox.setEnabled(False)
        self.brightness_checkbox.setEnabled(False)
        self.color_balance_checkbox.setEnabled(False)
        self.recent_list.setEnabled(False)
        self.device_combo.setEnabled(False)

        # 初始化进度
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在准备修复...")
        self.status_label.setText("正在准备修复...")
        self.log_message("开始修复流程")

        # 准备参数调节配置
        adjust_params = {
            'contrast': self.contrast_enhance,
            'brightness': self.brightness_adjust,
            'color': self.color_balance
        }

        # 快速修复
        if self.selected_algo == 0 and PREDICT_AVAILABLE:
            self.processing_thread = QThread()
            self.worker = RestorationWorker(
                image_paths=self.image_paths,
                mask_path=self.mask_path,
                output_dir=self.output_dir,
                format_type=self.format_combo.currentText(),
                model_path=self.lama_model_path,
                device=self.device,
                adjust_params=adjust_params
            )

            self.worker.moveToThread(self.processing_thread)
            self.processing_thread.started.connect(self.worker.run)
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.status_updated.connect(self.update_status)
            self.worker.log_updated.connect(self.log_message)
            self.worker.finished.connect(self.processing_finished)
            self.worker.finished.connect(self.worker.deleteLater)
            self.processing_thread.finished.connect(self.processing_thread.deleteLater)

            self.processing_thread.start()
            self.is_processing = True

        # 精细修复
        elif self.selected_algo == 1:
            self.process_fine_restoration(adjust_params)

    # ------------------------------
    # 取消修复
    # ------------------------------
    def cancel_processing(self):
        if self.is_processing and self.worker:
            self.worker.cancel()
            self.status_label.setText("正在取消修复...")
            self.log_message("用户请求取消修复任务")
        elif self.is_processing:
            self.status_label.setText("已取消修复")
            self.log_message("已取消修复任务")
            self.is_processing = False
            # 恢复UI
            self.cancel_btn.setEnabled(False)
            self.select_btn.setEnabled(True)
            self.fast_btn.setEnabled(True)
            self.fine_btn.setEnabled(True)
            self.contrast_checkbox.setEnabled(True)
            self.brightness_checkbox.setEnabled(True)
            self.color_balance_checkbox.setEnabled(True)
            self.prev_btn.setEnabled(len(self.image_paths) > 1)
            self.next_btn.setEnabled(len(self.image_paths) > 1)
            self.process_btn.setEnabled(True)
            self.recent_list.setEnabled(True)
            self.device_combo.setEnabled(True)

    # ------------------------------
    # 精细修复实现
    # ------------------------------
    def process_fine_restoration(self, adjust_params):
        try:
            start_time = time.time()
            current_path = self.image_paths[self.current_index]
            filename = os.path.splitext(os.path.basename(current_path))[0]
            output_format = self.format_combo.currentText().lower()
            output_path = os.path.join(self.output_dir, f"{filename}_restored.{output_format}")

            self.log_message(f"开始精细修复: {os.path.basename(current_path)}")
            self.progress_label.setText(f"修复中: {os.path.basename(current_path)}")

            # 读取图像和掩码
            src = cv.imread(current_path)
            mask = cv.imread(self.mask_path, 0)
            if src is None:
                raise ValueError(f"无法读取待修复图片: {current_path}")
            if mask is None:
                raise ValueError(f"无法读取掩码图片: {self.mask_path}")

            # 调整掩码尺寸
            if src.shape[:2] != mask.shape[:2]:
                mask = cv.resize(mask, (src.shape[1], src.shape[0]))
                self.log_message("掩码图像已调整为与原始图像相同尺寸")

            # 二值化掩码
            _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

            # 执行修复
            self.progress_bar.setValue(30)
            self.progress_label.setText("正在应用修复算法...")
            dst = cv.inpaint(src, mask, 3, cv.INPAINT_TELEA)

            # 应用参数调节
            if PARAM_ADJUST_AVAILABLE and self.param_adjuster:
                self.log_message(f"对精细修复结果应用参数调节")
                dst = self.param_adjuster.apply_all_adjustments(
                    dst,
                    apply_contrast=adjust_params['contrast'],
                    apply_brightness=adjust_params['brightness'],
                    apply_color=adjust_params['color']
                )

            # 保存结果
            self.progress_bar.setValue(80)
            self.progress_label.setText("正在保存结果...")
            cv.imwrite(output_path, dst)

            # 计算耗时
            elapsed_time = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

            # 更新统计
            self.stats_label.setText(f"""
                修复耗时: {elapsed_str}
                修复图像: 1/{len(self.image_paths)}
                平均区域: -- 像素
            """)

            # 存储结果并显示
            result_rgb = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
            self.restored_images = [result_rgb]
            self.restored_paths = [output_path]

            h, w, ch = result_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_image = qt_image.scaled(self.restored_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
            self.restored_preview.setPixmap(QPixmap.fromImage(scaled_image))
            self.restored_preview.setText("")

            # 完成状态
            self.progress_bar.setValue(100)
            self.progress_label.setText("修复完成")
            self.status_label.setText("修复完成")
            self.log_message(f"精细修复完成，结果已保存至: {output_path}")

            # 更新保存按钮
            self.save_single_btn.setEnabled(True)
            if len(self.image_paths) > 1:
                self.save_batch_btn.setEnabled(True)

        except Exception as e:
            self.log_message(f"修复过程出错: {str(e)}")
            self.status_label.setText(f"修复出错: {str(e)}")
            self.progress_label.setText("修复出错")

        finally:
            # 恢复UI
            self.cancel_btn.setEnabled(False)
            self.select_btn.setEnabled(True)
            self.fast_btn.setEnabled(True)
            self.fine_btn.setEnabled(True)
            self.contrast_checkbox.setEnabled(True)
            self.brightness_checkbox.setEnabled(True)
            self.color_balance_checkbox.setEnabled(True)
            self.prev_btn.setEnabled(len(self.image_paths) > 1)
            self.next_btn.setEnabled(len(self.image_paths) > 1)
            self.process_btn.setEnabled(True)
            self.recent_list.setEnabled(True)
            self.device_combo.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RelicRestorationPage()
    window.setWindowTitle("文物图像修复系统")
    window.resize(1200, 700)
    window.show()
    sys.exit(app.exec())