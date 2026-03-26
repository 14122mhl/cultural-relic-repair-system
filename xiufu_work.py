import sys
import os
import time
import json
import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QProgressBar, QListWidget, QGroupBox,
                             QRadioButton, QComboBox, QMessageBox, QFrame, QCheckBox,
                             QListWidgetItem, QApplication, QToolTip)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap, QImage


# 修复功能工作类（与界面分离）
class RestorationWorker(QObject):
    """修复功能工作类，与界面分离"""
    progress_updated = pyqtSignal(int, str)  # 进度值, 进度文本
    status_updated = pyqtSignal(str)  # 状态文本
    log_updated = pyqtSignal(str)  # 日志消息
    finished = pyqtSignal(object)  # 处理结果 (restored_images, stats) 或 None

    def __init__(self, images, image_paths, strength, algorithm, denoise_level,
                 detail_level, output_dir, save_format):
        super().__init__()
        self.images = [img.copy() for img in images]  # 原始图像列表（RGB格式）
        self.image_paths = image_paths  # 原始图像路径列表（用于保存时获取文件名）
        self.strength = strength  # 修复强度（0-10）
        self.algorithm = algorithm  # 算法类型：0-快速，1-精细（已删除深度修复）
        self.denoise_level = denoise_level  # 降噪程度（0-50）
        self.detail_level = detail_level  # 细节保留程度（0-100）
        self.output_dir = output_dir  # 输出目录
        self.save_format = save_format.lower()  # 保存格式（png/jpg/bmp）
        self.canceled = False  # 是否取消处理

    def run(self):
        """执行修复处理（核心逻辑）"""
        start_time = time.time()
        stats = {
            'time': '00:00:00',  # 总耗时
            'success_count': 0,  # 成功修复数量
            'total_count': len(self.images),  # 总处理数量
            'avg_damage_area': 0  # 平均损伤区域像素数
        }

        restored_images = []  # 修复后图像列表
        total_damage_area = 0  # 总损伤区域像素数

        try:
            # 验证输入参数
            if self.algorithm not in [0, 1]:
                raise ValueError("无效算法类型，仅支持0（快速修复）、1（精细修复）")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # 算法名称映射（已删除深度修复）
            algo_name_map = {0: "快速修复", 1: "精细修复"}
            self.log_updated.emit(f"开始{algo_name_map[self.algorithm]}，共 {stats['total_count']} 张图像")

            # 逐张处理图像
            for idx, (image, img_path) in enumerate(zip(self.images, self.image_paths)):
                if self.canceled:
                    self.log_updated.emit("修复任务已被用户取消")
                    self.finished.emit(None)
                    return

                # 1. 更新进度信息
                progress = int((idx + 1) / stats['total_count'] * 100)
                progress_text = f"处理中 {idx + 1}/{stats['total_count']} ({progress}%)"
                self.progress_updated.emit(progress, progress_text)
                self.status_updated.emit(f"修复第 {idx + 1} 张图像")
                self.log_updated.emit(f"开始处理：{os.path.basename(img_path)}")

                try:
                    # 2. 模拟损伤区域检测（可后续替换为真实算法）
                    damage_area = self.detect_damage_area(image)
                    total_damage_area += damage_area
                    self.log_updated.emit(f"检测到损伤区域：{damage_area} 像素")

                    # 3. 调用对应修复算法（已删除深度修复分支）
                    if self.algorithm == 0:
                        restored_img = self.quick_restoration(image)
                    else:  # self.algorithm == 1
                        restored_img = self.fine_restoration(image)

                    # 4. 保存修复结果（自动命名）
                    save_success = self.save_restored_image(restored_img, img_path, idx + 1)
                    if save_success:
                        restored_images.append(restored_img)
                        stats['success_count'] += 1
                        self.log_updated.emit(f"成功保存：{os.path.basename(img_path)}_restored.{self.save_format}")
                    else:
                        restored_images.append(None)
                        self.log_updated.emit(f"修复成功但保存失败：{os.path.basename(img_path)}")

                except Exception as e:
                    restored_images.append(None)
                    self.log_updated.emit(f"处理失败 [{os.path.basename(img_path)}]：{str(e)}")

            # 5. 计算统计信息
            if stats['success_count'] > 0:
                stats['avg_damage_area'] = int(total_damage_area / stats['success_count'])
            elapsed_time = int(time.time() - start_time)
            stats['time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

            # 6. 发送完成信号
            self.progress_updated.emit(100, f"修复完成！成功 {stats['success_count']}/{stats['total_count']} 张")
            self.status_updated.emit("修复任务已完成")
            self.log_updated.emit(
                f"修复总结：耗时{stats['time']}，成功{stats['success_count']}张，平均损伤区域{stats['avg_damage_area']}像素")
            self.finished.emit((restored_images, stats))

        except Exception as e:
            self.log_updated.emit(f"修复任务全局错误：{str(e)}")
            self.status_updated.emit("修复任务失败")
            self.finished.emit(None)

    def detect_damage_area(self, image):
        """模拟损伤区域检测（预留接口，可替换为真实算法）"""
        # 逻辑：根据修复强度动态生成模拟损伤区域大小（1%-4%图像像素）
        height, width = image.shape[:2]
        total_pixels = height * width
        damage_ratio = 0.01 + (self.strength / 10) * 0.03  # 1% ~ 4%
        return int(total_pixels * damage_ratio)

    def quick_restoration(self, image):
        """快速修复算法：轻量降噪+污点淡化（适合轻度损伤）"""
        # 转换为OpenCV支持的BGR格式
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. 轻量降噪（根据降噪等级调整）
        denoise_kernel = max(1, int(self.denoise_level / 10))  # 1-5
        if denoise_kernel % 2 == 0:
            denoise_kernel += 1  # 确保奇数核
        denoised = cv2.fastNlMeansDenoisingColored(
            bgr_img, None,
            h=self.denoise_level / 20,  # 降噪强度（0.0-2.5）
            hColor=self.denoise_level / 20,
            templateWindowSize=7,
            searchWindowSize=21
        )

        # 2. 中值滤波淡化污点（根据强度调整核大小）
        blur_kernel = 1 + 2 * int(self.strength / 2)  # 1-11（奇数）
        blurred = cv2.medianBlur(denoised, blur_kernel)

        # 3. 轻微锐化（避免过度模糊）
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        restored_bgr = cv2.filter2D(blurred, -1, sharpen_kernel)

        # 转回RGB格式（适配Qt显示）
        return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

    def fine_restoration(self, image):
        """精细修复算法：基于掩码的精准修复（适合中度损伤）"""
        # 转换为OpenCV支持的BGR格式
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. 生成模拟损伤掩码（预留：后续替换为真实损伤检测）
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # 假设高亮区域为损伤
        morph_kernel = np.ones((3, 3), np.uint8)
        damage_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)  # 闭合操作填充空洞

        # 2. 图像修复（Telea算法，适合小区域损伤）
        inpaint_radius = max(1, int(self.strength))  # 修复半径（1-10）
        inpainted = cv2.inpaint(
            bgr_img,
            damage_mask,
            inpaintRadius=inpaint_radius,
            flags=cv2.INPAINT_TELEA
        )

        # 3. 双边滤波保留细节（根据细节等级调整）
        detail_kernel = 1 + 2 * int(self.detail_level / 20)  # 1-11（奇数）
        detail_preserved = cv2.bilateralFilter(
            inpainted,
            d=detail_kernel,
            sigmaColor=75,  # 颜色相似度
            sigmaSpace=75  # 空间相似度
        )

        # 转回RGB格式（适配Qt显示）
        return cv2.cvtColor(detail_preserved, cv2.COLOR_BGR2RGB)

    def save_restored_image(self, image, original_path, index):
        """保存修复后的图像（自动生成文件名）"""
        try:
            # 生成保存路径：原始文件名_restored_序号.格式
            filename = os.path.splitext(os.path.basename(original_path))[0]
            save_filename = f"{filename}_restored_{index}.{self.save_format}"
            save_path = os.path.join(self.output_dir, save_filename)

            # 转换为BGR格式保存（OpenCV默认）
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr_image)
            return True

        except Exception as e:
            self.log_updated.emit(f"保存错误：{str(e)}")
            return False

    def cancel(self):
        """取消正在进行的修复任务"""
        self.canceled = True


# 文物图像修复界面类（与Worker联动）
class RelicRestorationPage(QWidget):
    """文物图像修复页面（含最近处理+修复功能联动）"""
    back_to_initial = pyqtSignal()

    def __init__(self):
        super().__init__()
        # 基础参数初始化
        self.original_images = []  # 原始图像数据（RGB格式）
        self.restored_images = []  # 修复后图像数据（RGB格式）
        self.image_paths = []  # 原始图像路径列表
        self.current_index = 0  # 当前显示图像索引
        self.output_dir = ""  # 输出目录
        self.processing_thread = None  # 修复处理线程
        self.worker = None  # 修复工作实例
        self.is_processing = False  # 是否正在处理

        # 算法选择（已删除深度修复，仅保留0-快速、1-精细）
        self.selected_algo = 0  # 0:快速修复, 1:精细修复
        self.algo_names = ["快速修复", "精细修复"]

        # 参数调节（预留接口，当前使用默认值，后续可扩展）
        self.repair_strength = 5  # 修复强度（0-10）
        self.denoise_level = 20  # 降噪程度（0-50）
        self.detail_level = 70  # 细节保留（0-100）

        # 最近文件相关
        self.recent_files = []  # 最近处理文件列表
        self.max_recent_files = 5  # 最多保留5个最近文件
        self.recent_file_path = "recent_files.json"  # 存储路径

        # 加载最近文件+初始化界面
        self.load_recent_files()
        self.init_ui()

    def init_ui(self):
        """初始化用户界面（与之前一致，确保布局完整）"""
        # 主布局（左-控制区，中-预览区，右-信息区）
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ---------------------- 左侧控制面板 ----------------------
        left_panel = QWidget()
        left_panel.setMinimumWidth(260)
        left_panel.setMaximumWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)

        # 1. 标题与返回按钮
        top_layout = QHBoxLayout()
        title_label = QLabel("文物图像修复")
        title_label.setFont(QFont("SimHei", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #1a365d;")

        back_btn = QPushButton("返回首页")
        back_btn.setFont(QFont("SimHei", 9))
        back_btn.setStyleSheet("""
            QPushButton {background-color: #3b82f6; color: white; border: none; border-radius: 4px; padding: 4px 8px;}
            QPushButton:hover {background-color: #2563eb;}
        """)
        back_btn.setMinimumHeight(25)
        back_btn.clicked.connect(self.back_to_initial.emit)

        top_layout.addWidget(title_label)
        top_layout.addStretch()
        top_layout.addWidget(back_btn)
        left_layout.addLayout(top_layout)

        # 2. 图像选择区域
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
        left_layout.addWidget(select_group)

        # 3. 算法选择区域（已删除深度修复选项）
        algo_group = QGroupBox("算法选择")
        algo_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.setContentsMargins(8, 8, 8, 8)
        algo_layout.setSpacing(6)

        # 快速修复选项
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
        self.fast_btn.setChecked(True)
        self.fast_btn.clicked.connect(lambda: self.set_selected_algo(0))

        fast_layout.addWidget(fast_label, 1)
        fast_layout.addWidget(self.fast_btn)
        algo_layout.addLayout(fast_layout)

        # 精细修复选项（已删除深度修复相关代码）
        fine_layout = QHBoxLayout()
        fine_label = QLabel("精细修复：中度损伤，细节好")
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

        # 4. 参数调节区域（预留接口，当前仅显示复选框，功能后续扩展）
        param_group = QGroupBox("参数调节")
        param_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        param_layout = QVBoxLayout(param_group)
        param_layout.setContentsMargins(8, 8, 8, 8)
        param_layout.setSpacing(8)

        # 对比度增强（预留）
        self.contrast_checkbox = QCheckBox("对比度增强")
        self.contrast_checkbox.setFont(QFont("SimHei", 9))
        self.contrast_checkbox.setStyleSheet("color: #4a5568;")
        self.contrast_checkbox.stateChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.contrast_checkbox)

        # 亮度调整（预留）
        self.brightness_checkbox = QCheckBox("亮度调整")
        self.brightness_checkbox.setFont(QFont("SimHei", 9))
        self.brightness_checkbox.setStyleSheet("color: #4a5568;")
        self.brightness_checkbox.stateChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.brightness_checkbox)

        # 色彩平衡（预留）
        self.color_balance_checkbox = QCheckBox("色彩平衡")
        self.color_balance_checkbox.setFont(QFont("SimHei", 9))
        self.color_balance_checkbox.setStyleSheet("color: #4a5568;")
        self.color_balance_checkbox.stateChanged.connect(self.on_param_changed)
        param_layout.addWidget(self.color_balance_checkbox)

        left_layout.addWidget(param_group)

        # 5. 最近处理区域
        recent_group = QGroupBox("最近处理")
        recent_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        recent_layout = QVBoxLayout(recent_group)
        recent_layout.setContentsMargins(8, 8, 8, 8)
        recent_layout.setSpacing(6)

        # 最近文件列表
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

        # 清空按钮
        clear_recent_btn = QPushButton("清空列表")
        clear_recent_btn.setFont(QFont("SimHei", 8))
        clear_recent_btn.setStyleSheet("""
            QPushButton {background-color: #e0e7ff; color: #4338ca; border: 1px solid #c7d2fe; border-radius: 3px; padding: 2px 6px;}
            QPushButton:hover {background-color: #c7d2fe;}
        """)
        clear_recent_btn.clicked.connect(self.clear_recent_files)
        recent_layout.addWidget(clear_recent_btn)

        left_layout.addWidget(recent_group)

        # 6. 输出设置区域
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

        # 7. 操作按钮区域
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

        # 取消修复按钮
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

        # 保存当前结果按钮
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

        # 批量保存结果按钮
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

        # ---------------------- 中间预览区域 ----------------------
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)

        # 标题与导航
        top_center_layout = QHBoxLayout()
        preview_label = QLabel("图像修复前后对比")
        preview_label.setFont(QFont("SimHei", 12, QFont.Weight.Bold))
        preview_label.setStyleSheet("color: #1a365d;")

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
        self.nav_widget.setVisible(False)

        top_center_layout.addWidget(preview_label)
        top_center_layout.addStretch()
        top_center_layout.addWidget(self.nav_widget)
        center_layout.addLayout(top_center_layout)

        # 图像预览容器
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

        # ---------------------- 右侧信息区域 ----------------------
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

        # 窗口基础设置
        self.setWindowTitle("文物图像修复")
        self.setMinimumSize(1200, 800)
        self.update_recent_list_display()  # 初始化最近文件列表

    # ---------------------- 最近文件相关功能 ----------------------
    def load_recent_files(self):
        """从JSON文件加载最近处理文件列表"""
        try:
            if os.path.exists(self.recent_file_path):
                with open(self.recent_file_path, 'r', encoding='utf-8') as f:
                    self.recent_files = json.load(f)
                # 过滤不存在的文件
                self.recent_files = [p for p in self.recent_files if os.path.exists(p)]
                self.save_recent_files()
        except Exception as e:
            self.log_message(f"加载最近文件失败：{str(e)}")
            self.recent_files = []

    def save_recent_files(self):
        """将最近处理文件列表保存到JSON文件"""
        try:
            with open(self.recent_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.recent_files[:self.max_recent_files], f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log_message(f"保存最近文件失败：{str(e)}")

    def add_recent_file(self, file_path):
        """添加文件到最近列表（去重+限制数量）"""
        if not file_path or not os.path.exists(file_path):
            return
        # 去重（移旧添新）
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        # 限制数量
        if len(self.recent_files) > self.max_recent_files:
            self.recent_files = self.recent_files[:self.max_recent_files]
        # 保存并更新UI
        self.save_recent_files()
        self.update_recent_list_display()

    def update_recent_list_display(self):
        """更新最近文件列表UI"""
        self.recent_list.clear()
        for path in self.recent_files:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.ToolTipRole, path)  # 悬停显示完整路径
            self.recent_list.addItem(item)

    def clear_recent_files(self):
        """清空最近文件列表"""
        self.recent_files = []
        self.save_recent_files()
        self.update_recent_list_display()
        self.log_message("已清空最近处理文件列表")

    def on_recent_item_clicked(self, item):
        """点击最近文件列表项，加载对应文件"""
        file_path = item.data(Qt.ItemDataRole.ToolTipRole)
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "文件不存在", f"文件已移动或删除：\n{file_path}")
            self.recent_files.remove(file_path)
            self.save_recent_files()
            self.update_recent_list_display()
            return

        # 区分文件/文件夹加载
        if os.path.isfile(file_path):
            self.single_radio.setChecked(True)
            self.load_single_image(file_path)
        else:
            self.batch_radio.setChecked(True)
            self.load_batch_images(file_path)
        # 更新最近列表（移至顶部）
        self.add_recent_file(file_path)

    # ---------------------- 图像加载相关功能 ----------------------
    def select_images(self):
        """选择单张/批量图像"""
        if self.single_radio.isChecked():
            # 单张选择
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择单张图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
            )
            if file_path:
                self.load_single_image(file_path)
                self.add_recent_file(file_path)
        else:
            # 批量选择（文件夹）
            dir_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
            if dir_path:
                self.load_batch_images(dir_path)
                self.add_recent_file(dir_path)

    def load_single_image(self, file_path):
        """加载单张图像（RGB格式）"""
        try:
            # 用OpenCV读取图像并转换为RGB
            bgr_img = cv2.imread(file_path)
            if bgr_img is None:
                raise ValueError("无法读取图像文件")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # 更新数据与UI
            self.image_paths = [file_path]
            self.original_images = [rgb_img]
            self.restored_images = []
            self.current_index = 0

            # 更新预览
            self.update_image_preview()
            self.path_label.setText(f"已选择：{os.path.basename(file_path)}")
            self.nav_widget.setVisible(False)
            self.process_btn.setEnabled(bool(self.output_dir))
            self.save_single_btn.setEnabled(False)
            self.save_batch_btn.setEnabled(False)
            self.log_message(f"已加载单张图像：{os.path.basename(file_path)}")

        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"单张图像加载错误：{str(e)}")

    def load_batch_images(self, dir_path):
        """加载文件夹内所有图像（RGB格式）"""
        try:
            # 筛选图像文件
            img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
            file_paths = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if os.path.splitext(f)[1].lower() in img_extensions
            ]
            if not file_paths:
                raise ValueError("文件夹内无有效图像文件")

            # 批量读取并转换为RGB
            original_images = []
            valid_paths = []
            for path in file_paths:
                bgr_img = cv2.imread(path)
                if bgr_img is not None:
                    original_images.append(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
                    valid_paths.append(path)

            if not valid_paths:
                raise ValueError("所有图像文件均无法读取")

            # 更新数据与UI
            self.image_paths = valid_paths
            self.original_images = original_images
            self.restored_images = [None] * len(original_images)
            self.current_index = 0

            # 更新预览
            self.update_image_preview()
            self.path_label.setText(f"已选择：{len(valid_paths)} 张图像（{os.path.basename(dir_path)}）")
            self.nav_widget.setVisible(len(valid_paths) > 1)
            self.update_navigation()
            self.process_btn.setEnabled(bool(self.output_dir))
            self.save_single_btn.setEnabled(False)
            self.save_batch_btn.setEnabled(False)
            self.log_message(f"已加载批量图像：{len(valid_paths)} 张（{os.path.basename(dir_path)}）")

        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"批量图像加载错误：{str(e)}")

    def update_image_preview(self):
        """更新图像预览（修复前+修复后）"""
        # 修复前预览
        if self.original_images:
            rgb_img = self.original_images[self.current_index]
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.original_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.original_preview.setPixmap(pixmap)
        else:
            self.original_preview.setText("请选择图像")
            self.original_preview.setPixmap(QPixmap())

        # 修复后预览
        if self.restored_images and self.current_index < len(self.restored_images):
            restored_img = self.restored_images[self.current_index]
            if restored_img is not None:
                h, w, ch = restored_img.shape
                bytes_per_line = ch * w
                q_img = QImage(restored_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img).scaled(
                    self.restored_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.restored_preview.setPixmap(pixmap)
            else:
                self.restored_preview.setText("修复失败")
        else:
            self.restored_preview.setText("尚未修复")
            self.restored_preview.setPixmap(QPixmap())

    # ---------------------- 导航与算法选择 ----------------------
    def update_navigation(self):
        """更新批量导航按钮状态"""
        total = len(self.image_paths)
        if total <= 1:
            self.nav_widget.setVisible(False)
            return
        self.nav_widget.setVisible(True)
        self.image_index_label.setText(f"{self.current_index + 1}/{total}")
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < total - 1)

    def show_prev_image(self):
        """显示上一张图像"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_navigation()
            self.update_image_preview()

    def show_next_image(self):
        """显示下一张图像"""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_navigation()
            self.update_image_preview()

    def set_selected_algo(self, index):
        """设置选中的算法（已删除深度修复相关）"""
        if index not in [0, 1]:
            return
        self.selected_algo = index

        # 更新按钮样式（取消深度修复按钮样式）
        self.fast_btn.setStyleSheet("""
            QPushButton {background-color: #94a3b8; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
            QPushButton:hover {background-color: #64748b;}
        """)
        self.fine_btn.setStyleSheet("""
            QPushButton {background-color: #94a3b8; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
            QPushButton:hover {background-color: #64748b;}
        """)

        # 激活选中算法按钮
        if index == 0:
            self.fast_btn.setStyleSheet("""
                QPushButton {background-color: #3b82f6; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
                QPushButton:hover {background-color: #2563eb;}
            """)
        else:
            self.fine_btn.setStyleSheet("""
                QPushButton {background-color: #3b82f6; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
                QPushButton:hover {background-color: #2563eb;}
            """)

        self.log_message(f"已选择算法：{self.algo_names[self.selected_algo]}")

    # ---------------------- 修复处理相关功能 ----------------------
    def select_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择修复结果保存目录")
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
            self.process_btn.setEnabled(bool(self.image_paths))

    def start_processing(self):
        """启动修复处理（与Worker联动）"""
        if not self.image_paths or not self.output_dir:
            QMessageBox.warning(self, "参数不全", "请先选择图像和输出目录")
            return
        if self.is_processing:
            QMessageBox.information(self, "正在处理", "已有修复任务正在进行中")
            return

        # 1. 初始化Worker和线程
        self.worker = RestorationWorker(
            images=self.original_images,
            image_paths=self.image_paths,
            strength=self.repair_strength,
            algorithm=self.selected_algo,
            denoise_level=self.denoise_level,
            detail_level=self.detail_level,
            output_dir=self.output_dir,
            save_format=self.format_combo.currentText()
        )
        self.processing_thread = QThread()
        self.worker.moveToThread(self.processing_thread)

        # 2. 连接Worker信号与界面槽函数
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.status_updated.connect(self.on_status_updated)
        self.worker.log_updated.connect(self.log_message)
        self.worker.finished.connect(self.on_process_finished)
        self.processing_thread.started.connect(self.worker.run)

        # 3. 更新UI状态
        self.is_processing = True
        self.process_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.recent_list.setEnabled(False)
        self.fast_btn.setEnabled(False)
        self.fine_btn.setEnabled(False)
        self.contrast_checkbox.setEnabled(False)
        self.brightness_checkbox.setEnabled(False)
        self.color_balance_checkbox.setEnabled(False)

        # 4. 启动线程
        self.processing_thread.start()
        self.log_message(f"启动{self.algo_names[self.selected_algo]}任务，输出目录：{os.path.basename(self.output_dir)}")

    def cancel_processing(self):
        """取消修复处理"""
        if not self.is_processing or not self.worker:
            return
        if QMessageBox.question(self, "确认取消", "是否要取消当前修复任务？",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No:
            return

        # 通知Worker取消任务
        self.worker.cancel()
        self.status_label.setText("正在取消任务...")
        self.log_message("用户已发起取消请求，等待任务终止")

    def on_progress_updated(self, progress, text):
        """更新修复进度UI"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(text)

    def on_status_updated(self, status):
        """更新状态标签"""
        self.status_label.setText(status)

    def on_process_finished(self, result):
        """修复任务完成后的处理"""
        # 1. 清理线程资源
        self.processing_thread.quit()
        self.processing_thread.wait()
        self.processing_thread = None
        self.worker = None

        # 2. 更新UI状态
        self.is_processing = False
        self.select_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.recent_list.setEnabled(True)
        self.fast_btn.setEnabled(True)
        self.fine_btn.setEnabled(True)
        self.contrast_checkbox.setEnabled(True)
        self.brightness_checkbox.setEnabled(True)
        self.color_balance_checkbox.setEnabled(True)
        self.process_btn.setEnabled(True)

        # 3. 处理修复结果
        if result is None:
            self.status_label.setText("修复任务未完成")
            return

        restored_images, stats = result
        self.restored_images = restored_images

        # 更新统计信息UI
        self.stats_label.setText(f"""
            修复耗时: {stats['time']}
            修复图像: {stats['success_count']}/{stats['total_count']}
            平均区域: {stats['avg_damage_area']} 像素
        """)

        # 更新预览（显示第一张修复结果）
        if restored_images and restored_images[0] is not None:
            self.current_index = 0
            self.update_image_preview()
            self.save_single_btn.setEnabled(True)
            self.save_batch_btn.setEnabled(stats['success_count'] > 1)

    # ---------------------- 保存与参数相关 ----------------------
    def save_current_restored_image(self):
        """保存当前显示的修复结果（单独保存）"""
        if not self.restored_images or self.current_index >= len(self.restored_images):
            QMessageBox.warning(self, "无结果", "无可用的修复结果可保存")
            return

        restored_img = self.restored_images[self.current_index]
        if restored_img is None:
            QMessageBox.warning(self, "保存失败", "当前图像修复失败，无法保存")
            return

        # 选择保存路径
        original_path = self.image_paths[self.current_index]
        filename = os.path.splitext(os.path.basename(original_path))[0] + "_restored"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存当前修复结果",
            os.path.join(self.output_dir, filename),
            f"{self.format_combo.currentText()}文件 (*.{self.format_combo.currentText().lower()})"
        )
        if not save_path:
            return

        # 保存图像（RGB转BGR）
        try:
            bgr_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr_img)
            QMessageBox.information(self, "保存成功", f"已保存到：\n{os.path.basename(save_path)}")
            self.log_message(f"手动保存当前结果：{os.path.basename(save_path)}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存错误：{str(e)}")

    def batch_save_restored_images(self):
        """批量保存所有修复结果（补充保存）"""
        if not self.restored_images or len(self.restored_images) == 0:
            QMessageBox.warning(self, "无结果", "无可用的修复结果可批量保存")
            return

        # 选择批量保存目录
        save_dir = QFileDialog.getExistingDirectory(self, "选择批量保存目录")
        if not save_dir:
            return

        # 批量保存
        success_count = 0
        for idx, (restored_img, img_path) in enumerate(zip(self.restored_images, self.image_paths)):
            if restored_img is None:
                continue
            try:
                # 生成文件名
                filename = os.path.splitext(os.path.basename(img_path))[
                               0] + f"_batch_restored_{idx + 1}.{self.format_combo.currentText().lower()}"
                save_path = os.path.join(save_dir, filename)
                # 保存（RGB转BGR）
                bgr_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_img)
                success_count += 1
            except Exception as e:
                self.log_message(f"批量保存失败 [{idx + 1}张]：{str(e)}")

        # 提示结果
        QMessageBox.information(self, "批量保存完成", f"共保存 {success_count}/{len(self.restored_images)} 张修复结果")
        self.log_message(f"批量保存完成：成功{success_count}张，目录：{os.path.basename(save_dir)}")

    def on_param_changed(self, state):
        """参数调节复选框状态变化（预留接口，当前仅日志）"""
        param_name = ""
        if self.sender() == self.contrast_checkbox:
            param_name = "对比度增强"
        elif self.sender() == self.brightness_checkbox:
            param_name = "亮度调整"
        elif self.sender() == self.color_balance_checkbox:
            param_name = "色彩平衡"

        status = "开启" if state == Qt.CheckState.Checked.value else "关闭"
        self.log_message(f"{param_name}：{status}（功能后续扩展）")

    def log_message(self, message):
        """添加处理日志"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_list.addItem(f"[{timestamp}] {message}")
        self.log_list.scrollToBottom()

    # ---------------------- 窗口关闭处理 ----------------------
    def closeEvent(self, event):
        """窗口关闭时保存最近文件并终止任务"""
        # 保存最近文件
        self.save_recent_files()
        # 终止正在进行的任务
        if self.is_processing and self.worker:
            self.worker.cancel()
            if self.processing_thread:
                self.processing_thread.quit()
                self.processing_thread.wait()
        event.accept()


# ---------------------- 主程序入口 ----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 确保中文显示正常
    app.setFont(QFont("SimHei"))
    # 启动修复界面
    window = RelicRestorationPage()
    window.show()
    sys.exit(app.exec())