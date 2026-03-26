import sys
import os
import random
import time
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QProgressBar, QListWidget, QGroupBox,
                             QRadioButton, QCheckBox, QSlider, QSpinBox, QComboBox,
                             QGridLayout, QMessageBox, QFrame, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QColor

import numpy as np
import cv2


class Dat(QWidget):
    """数据集生成页面，用于给正常图像添加各种损失"""
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
        """初始化用户界面"""
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

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

        # 图像选择区域
        select_group = QGroupBox("图像选择")
        select_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        select_layout = QVBoxLayout(select_group)
        select_layout.setSpacing(10)

        # 单张/批量选择
        self.single_radio = QRadioButton("单张处理")
        self.batch_radio = QRadioButton("批量处理")
        self.single_radio.setChecked(True)
        self.single_radio.setFont(QFont("SimHei", 10))
        self.batch_radio.setFont(QFont("SimHei", 10))

        select_layout.addWidget(self.single_radio)
        select_layout.addWidget(self.batch_radio)

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

        # 损失类型选择
        damage_group = QGroupBox("损失类型")
        damage_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        damage_layout = QVBoxLayout(damage_group)
        damage_layout.setSpacing(10)

        # 条状损失
        self.strip_check = QCheckBox("条状损失")
        self.strip_check.setChecked(True)
        self.strip_check.setFont(QFont("SimHei", 10))
        damage_layout.addWidget(self.strip_check)

        # 条状损失参数
        strip_params = QWidget()
        strip_params_layout = QGridLayout(strip_params)
        strip_params_layout.setContentsMargins(10, 5, 10, 5)
        strip_params_layout.setSpacing(8)

        # 数量
        strip_params_layout.addWidget(QLabel("数量:"), 0, 0)
        self.strip_count = QSpinBox()
        self.strip_count.setRange(1, 20)
        self.strip_count.setValue(3)
        strip_params_layout.addWidget(self.strip_count, 0, 1)

        # 宽度
        strip_params_layout.addWidget(QLabel("宽度:"), 1, 0)
        self.strip_width = QSpinBox()
        self.strip_width.setRange(1, 50)
        self.strip_width.setValue(10)
        strip_params_layout.addWidget(self.strip_width, 1, 1)

        # 方向
        strip_params_layout.addWidget(QLabel("方向:"), 2, 0)
        self.strip_dir = QComboBox()
        self.strip_dir.addItems(["随机", "水平", "垂直"])
        strip_params_layout.addWidget(self.strip_dir, 2, 1)

        damage_layout.addWidget(strip_params)

        # 块状损失
        self.block_check = QCheckBox("块状损失")
        self.block_check.setChecked(True)
        self.block_check.setFont(QFont("SimHei", 10))
        damage_layout.addWidget(self.block_check)

        # 块状损失参数
        block_params = QWidget()
        block_params_layout = QGridLayout(block_params)
        block_params_layout.setContentsMargins(10, 5, 10, 5)
        block_params_layout.setSpacing(8)

        # 数量
        block_params_layout.addWidget(QLabel("数量:"), 0, 0)
        self.block_count = QSpinBox()
        self.block_count.setRange(1, 15)
        self.block_count.setValue(2)
        block_params_layout.addWidget(self.block_count, 0, 1)

        # 大小
        block_params_layout.addWidget(QLabel("大小:"), 1, 0)
        self.block_size = QSpinBox()
        self.block_size.setRange(10, 200)
        self.block_size.setValue(50)
        block_params_layout.addWidget(self.block_size, 1, 1)

        damage_layout.addWidget(block_params)

        left_layout.addWidget(damage_group)

        # 输出设置
        output_group = QGroupBox("输出设置")
        output_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        output_layout = QVBoxLayout(output_group)
        output_layout.setSpacing(10)

        # 输出目录
        output_dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel("未选择目录")
        self.output_dir_label.setFont(QFont("SimHei", 9))
        self.output_dir_label.setStyleSheet(
            "color: #4a5568; border: 1px solid #e2e8f0; padding: 5px; border-radius: 4px; flex: 1;")
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
        format_layout.addWidget(QLabel("保存格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "BMP"])
        self.format_combo.setCurrentText("PNG")
        format_layout.addWidget(self.format_combo)
        output_layout.addLayout(format_layout)

        left_layout.addWidget(output_group)

        # 处理按钮
        self.process_btn = QPushButton("开始处理")
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
        left_layout.addWidget(self.process_btn)

        # 取消按钮
        self.cancel_btn = QPushButton("取消处理")
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
        left_layout.addWidget(self.cancel_btn)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # 中间预览区域
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(10)

        # 预览标题
        preview_label = QLabel("图像预览")
        preview_label.setFont(QFont("SimHei", 12, QFont.Weight.Bold))
        preview_label.setStyleSheet("color: #1a365d;")
        center_layout.addWidget(preview_label)

        # 图像显示区域
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        # 原图显示
        self.original_preview = QLabel("原始图像")
        self.original_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_preview.setMinimumSize(400, 300)
        self.original_preview.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; color: #64748b;")
        preview_layout.addWidget(self.original_preview)

        # 处理后图像显示
        self.processed_preview = QLabel("处理后图像")
        self.processed_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_preview.setMinimumSize(400, 300)
        self.processed_preview.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; color: #64748b;")
        preview_layout.addWidget(self.processed_preview)

        center_layout.addWidget(preview_widget, 1)

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

        # 右侧进度和日志区域
        right_panel = QWidget()
        right_panel.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)

        # 进度显示
        progress_group = QGroupBox("处理进度")
        progress_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("等待处理...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("SimHei", 10))
        self.status_label.setStyleSheet("color: #4a5568;")
        progress_layout.addWidget(self.status_label)

        right_layout.addWidget(progress_group)

        # 处理日志
        log_group = QGroupBox("处理日志")
        log_group.setFont(QFont("SimHei", 11, QFont.Weight.Bold))
        log_layout = QVBoxLayout(log_group)

        self.log_list = QListWidget()
        self.log_list.setFont(QFont("SimHei", 9))
        log_layout.addWidget(self.log_list)

        right_layout.addWidget(log_group, 1)

        main_layout.addWidget(right_panel)

    def select_images(self):
        """选择单张或多张图像"""
        if self.single_radio.isChecked():
            # 单张处理
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
            )

            if file_path:
                self.image_path = file_path
                self.image_paths = [file_path]
                self.path_label.setText(f"已选择: {os.path.basename(file_path)}")

                # 加载并显示图像
                self.load_image(file_path)
                self.process_btn.setEnabled(True if self.output_dir else False)
                self.refresh_preview_btn.setEnabled(True)

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
                    self.path_label.setText(f"已选择文件夹: {dir_path}\n包含 {len(self.image_paths)} 张图像")

                    # 预览第一张图像
                    self.load_image(self.image_paths[0])
                    self.process_btn.setEnabled(True if self.output_dir else False)
                    self.refresh_preview_btn.setEnabled(True)
                else:
                    QMessageBox.warning(self, "警告", "所选文件夹中未找到图像文件")
                    self.path_label.setText("未选择任何文件")

    def load_image(self, file_path):
        """加载图像并显示预览"""
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise Exception("无法读取图像文件")

            # 转换为RGB格式
            self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 显示原图预览
            self.display_image(self.original_preview, self.original_image)

            # 生成处理后的预览图
            self.refresh_preview()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
            self.log_message(f"加载图像失败: {str(e)}")

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

    def add_strip_damage(self, image):
        """添加条状损失"""
        if not self.strip_check.isChecked():
            return image

        img = image.copy()
        height, width = img.shape[:2]
        count = self.strip_count.value()
        strip_width = self.strip_width.value()
        direction = self.strip_dir.currentText()

        for _ in range(count):
            # 随机选择方向（如果设置为随机）
            if direction == "随机":
                is_horizontal = random.choice([True, False])
            else:
                is_horizontal = (direction == "水平")

            if is_horizontal:
                # 水平条
                y = random.randint(0, height - 1)
                cv2.rectangle(
                    img,
                    (0, y),
                    (width, min(y + strip_width, height - 1)),
                    (0, 0, 0),  # 黑色损失
                    -1
                )
            else:
                # 垂直条
                x = random.randint(0, width - 1)
                cv2.rectangle(
                    img,
                    (x, 0),
                    (min(x + strip_width, width - 1), height),
                    (0, 0, 0),  # 黑色损失
                    -1
                )

        return img

    def add_block_damage(self, image):
        """添加块状损失"""
        if not self.block_check.isChecked():
            return image

        img = image.copy()
        height, width = img.shape[:2]
        count = self.block_count.value()
        block_size = self.block_size.value()

        for _ in range(count):
            # 随机位置
            x = random.randint(0, width - block_size)
            y = random.randint(0, height - block_size)

            # 随机块大小（在设置值的50%-100%之间）
            size = random.randint(int(block_size * 0.5), block_size)

            # 确保块不超出图像边界
            x2 = min(x + size, width - 1)
            y2 = min(y + size, height - 1)

            cv2.rectangle(
                img,
                (x, y),
                (x2, y2),
                (0, 0, 0),  # 黑色损失
                -1
            )

        return img

    def process_image(self, image):
        """处理图像，添加所有选中的损失类型"""
        if image is None:
            return None

        # 添加条状损失
        result = self.add_strip_damage(image)

        # 添加块状损失
        result = self.add_block_damage(result)

        return result

    def refresh_preview(self):
        """刷新预览图像"""
        if self.original_image is None:
            return

        # 处理图像
        self.processed_image = self.process_image(self.original_image)

        # 显示处理后的图像
        self.display_image(self.processed_preview, self.processed_image)

        self.log_message("已更新预览图像")

    def start_processing(self):
        """开始处理图像"""
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请先选择图像")
            return

        if not self.output_dir:
            QMessageBox.warning(self, "警告", "请先选择输出目录")
            return

        if not (self.strip_check.isChecked() or self.block_check.isChecked()):
            QMessageBox.warning(self, "警告", "请至少选择一种损失类型")
            return

        # 禁用相关按钮
        self.process_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.refresh_preview_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.single_radio.setEnabled(False)
        self.batch_radio.setEnabled(False)

        # 创建并启动处理线程
        self.processing_thread = ProcessingThread(
            self.image_paths,
            self.output_dir,
            self.process_image,
            self.format_combo.currentText()
        )

        # 连接信号
        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.status_updated.connect(self.status_label.setText)
        self.processing_thread.log_updated.connect(self.log_message)
        self.processing_thread.finished.connect(self.processing_finished)

        # 开始处理
        self.is_processing = True
        self.processing_thread.start()

    def cancel_processing(self):
        """取消处理"""
        if self.processing_thread and self.is_processing:
            self.processing_thread.cancel()
            self.status_label.setText("正在取消处理...")
            self.log_message("用户取消了处理")

    def processing_finished(self, success):
        """处理完成回调"""
        self.is_processing = False
        self.cancel_btn.setEnabled(False)
        self.select_btn.setEnabled(True)
        self.single_radio.setEnabled(True)
        self.batch_radio.setEnabled(True)

        if self.image_paths:
            self.process_btn.setEnabled(True)
            self.refresh_preview_btn.setEnabled(True)

        if success:
            self.status_label.setText("处理完成")
            QMessageBox.information(self, "完成", "图像处理已完成")
        else:
            self.status_label.setText("处理已取消")

    def log_message(self, message):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_list.addItem(f"[{timestamp}] {message}")
        self.log_list.scrollToBottom()  # 滚动到最新条目


class ProcessingThread(QThread):
    """图像处理线程，避免界面卡顿"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool)  # 处理完成信号，参数表示是否成功完成

    def __init__(self, image_paths, output_dir, process_func, output_format):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.process_func = process_func  # 处理函数
        self.output_format = output_format.lower()
        self.canceled = False

    def run(self):
        """线程运行函数"""
        total = len(self.image_paths)

        for i, file_path in enumerate(self.image_paths):
            if self.canceled:
                self.finished.emit(False)
                return

            try:
                # 更新进度
                progress = int((i + 1) / total * 100)
                self.progress_updated.emit(progress)
                self.status_updated.emit(f"正在处理 {i + 1}/{total}")

                # 读取图像
                image = cv2.imread(file_path)
                if image is None:
                    self.log_updated.emit(f"无法读取图像: {os.path.basename(file_path)}")
                    continue

                # 转换为RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 处理图像
                processed_image = self.process_func(rgb_image)
                if processed_image is None:
                    self.log_updated.emit(f"处理失败: {os.path.basename(file_path)}")
                    continue

                # 转换回BGR用于保存
                bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

                # 构建输出路径
                filename = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(
                    self.output_dir,
                    f"{filename}_damaged.{self.output_format}"
                )

                # 保存图像
                cv2.imwrite(output_path, bgr_image)
                self.log_updated.emit(f"已保存: {os.path.basename(output_path)}")

            except Exception as e:
                self.log_updated.emit(f"处理错误 {os.path.basename(file_path)}: {str(e)}")

        self.progress_updated.emit(100)
        self.finished.emit(True)

    def cancel(self):
        """取消处理"""
        self.canceled = True


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    # 确保中文显示正常
    font = QFont("SimHei")
    app.setFont(font)

    window = DatasetGeneratorPage()
    window.setWindowTitle("数据集生成")
    window.resize(1200, 800)
    window.show()

    sys.exit(app.exec())
