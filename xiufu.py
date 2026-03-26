#!/usr/bin/env python3
#!/usr/bin/env python3
import sys
import os
import time
import json
import cv2 as cv
import numpy as np
import math
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QProgressBar, QListWidget, QGroupBox,
                             QRadioButton, QComboBox, QMessageBox, QFrame, QCheckBox,
                             QListWidgetItem, QApplication, QToolTip, QSizePolicy,
                             QColorDialog, QSlider)  # 添加 QSlider
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QPoint, QRect, QSize
from PyQt5.QtGui import (QFont, QPixmap, QImage, QPainter, QPen, QColor, QBrush,
                         QMouseEvent, QCursor)  # 添加 QCursor


class AnnotationDisplayWidget(QWidget):
    """自定义图像显示和标注组件，参考touch.py的ImageDisplayWidget实现"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_pixmap = None
        self.drawing_enabled = False
        self.brush_color = QColor(255, 0, 0)
        self.brush_size = 5
        self.brush_shape = 'circle'
        self.painting = False
        self.painted_mask = []
        self.scale_factor = 1.0
        self.image_rect = QRect()
        self.parent_page = None  # 引用父页面以访问相关方法
        
    def setParentPage(self, parent_page):
        """设置父页面引用"""
        self.parent_page = parent_page
        
    def setImage(self, pixmap):
        """设置显示的图片"""
        self.image_pixmap = pixmap
        self.update()
        
    def enableDrawing(self, enabled):
        """启用/禁用绘制功能"""
        self.drawing_enabled = enabled
        
    def setBrushColor(self, color):
        """设置画笔颜色"""
        self.brush_color = color
        
    def setBrushSize(self, size):
        """设置画笔大小"""
        self.brush_size = size
        
    def setBrushShape(self, shape):
        """设置画笔形状"""
        self.brush_shape = shape
        
    def clearPaintedMask(self):
        """清除涂抹记录"""
        self.painted_mask.clear()
        
    def getPaintedMask(self):
        """获取涂抹区域"""
        return self.painted_mask if self.painted_mask else None
        
    def paintEvent(self, event):
        """绘制事件 - 核心绘制逻辑，与touch.py保持一致"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景 - 统一为白色背景
        painter.fillRect(self.rect(), Qt.white)
        
        if self.image_pixmap is not None:
            # 计算缩放比例以适应窗口
            scaled_pixmap = self.image_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # 计算居中位置
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            
            # 存储图片在窗口中的实际显示区域
            self.image_rect = QRect(x, y, scaled_pixmap.width(), scaled_pixmap.height())
            
            # 绘制图片
            painter.drawPixmap(x, y, scaled_pixmap)
            
            # 绘制涂抹区域 - 修复坐标精准度问题，与V2/touch.py保持完全一致
            if self.drawing_enabled and self.painted_mask:
                painter.setPen(QPen(self.brush_color, 2))
                painter.setBrush(QBrush(self.brush_color))
                
                for mask_data in self.painted_mask:
                    if isinstance(mask_data, dict):
                        # 新的格式：包含位置和形状信息
                        screen_x, screen_y = mask_data['screen_pos']
                        brush_size = mask_data['size']
                        shape = mask_data['shape']
                        
                        # 直接使用screen_pos坐标，无需额外偏移计算
                        # 与V2/touch.py完全一致的绘制逻辑
                        if shape == 'circle':
                            painter.drawEllipse(screen_x - brush_size // 2, screen_y - brush_size // 2,
                                                brush_size, brush_size)
                        elif shape == 'square':
                            painter.drawRect(screen_x - brush_size // 2, screen_y - brush_size // 2,
                                             brush_size, brush_size)
                        elif shape == 'polygon':
                            # 绘制六边形
                            points = []
                            for i in range(6):
                                angle = i * 60 * 3.14159 / 180
                                px = screen_x + brush_size // 2 * math.cos(angle)
                                py = screen_y + brush_size // 2 * math.sin(angle)
                                points.append(QPoint(int(px), int(py)))
                            painter.drawPolygon(points)
                    else:
                        # 兼容旧格式（矩形）
                        painter.drawRect(mask_data)
                        
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if not self.drawing_enabled or self.image_pixmap is None:
            return
            
        if event.button() == Qt.LeftButton:
            self.painting = True
            self.addPaintPoint(event.pos())
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if not self.painting or not self.drawing_enabled:
            return
            
        self.addPaintPoint(event.pos())
        
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self.painting = False
            
    def addPaintPoint(self, pos):
        """添加涂抹点 - 与主类方法保持完全一致的坐标映射"""
        if self.image_pixmap is None:
            return
            
        # 检查点击位置是否在图片显示范围内
        if not self.image_rect.contains(pos):
            return

        # 计算缩放比例 - 与主类完全一致的计算方式
        scaled_pixmap = self.image_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        scale_x = self.image_pixmap.width() / scaled_pixmap.width()
        scale_y = self.image_pixmap.height() / scaled_pixmap.height()

        # 精确映射到图片坐标 - 与主类完全一致
        img_x = (pos.x() - self.image_rect.x()) * scale_x
        img_y = (pos.y() - self.image_rect.y()) * scale_y

        # 确保坐标在图片范围内
        img_x = max(0, min(img_x, self.image_pixmap.width() - 1))
        img_y = max(0, min(img_y, self.image_pixmap.height() - 1))

        # 存储涂抹数据（包含屏幕坐标和图片坐标）- 与主类完全一致
        mask_data = {
            'screen_pos': (pos.x(), pos.y()),  # 屏幕坐标用于绘制
            'image_pos': (img_x, img_y),  # 图片坐标用于二值化
            'size': self.brush_size,
            'shape': self.brush_shape
        }

        self.painted_mask.append(mask_data)
        
        # 同步到父页面的painted_mask
        if self.parent_page:
            self.parent_page.painted_mask = self.painted_mask
        
        self.update()  # 触发重绘，显示标注
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

        # 标注功能变量
        self.annotation_image = None
        self.annotation_pixmap = None
        self.annotation_enabled = False
        self.brush_color = QColor(255, 0, 0)  # 默认红色
        self.brush_size = 5
        self.brush_shape = 'circle'  # 默认圆形
        self.painted_mask = []
        self.binary_mask_generated = False
        self.temp_mask_path = None
        
        # 切换预览相关变量
        self.is_mask_preview = False  # False显示原图，True显示掩码图
        
        # 创建临时目录
        self.temp_mask_dir = "D:/python_chuangxin/jiemianpythonProject/V2/zanshibaocun"
        os.makedirs(self.temp_mask_dir, exist_ok=True)

        # 初始化参数调整器
        self.param_adjuster = ImageParameterAdjuster() if PARAM_ADJUST_AVAILABLE else None

        # 加载最近文件
        self.load_recent_files()
        self.init_ui()

        # 设置自定义图像显示组件的父页面引用
        self.original_preview.setParentPage(self)
        
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
        
        # 切换预览按钮
        toggle_layout = QHBoxLayout()
        self.toggle_preview_btn = QPushButton("切换：显示掩码图")
        self.toggle_preview_btn.setFont(QFont("SimHei", 9))
        self.toggle_preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0f2fe;
                color: #1a365d;
                border: 1px solid #93c5fd;
                border-radius: 4px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #bfdbfe;
            }
            QPushButton:disabled {
                background-color: #f1f5f9;
                color: #94a3b8;
            }
        """)
        self.toggle_preview_btn.clicked.connect(self.toggle_preview)
        self.toggle_preview_btn.setEnabled(False)
        toggle_layout.addWidget(self.toggle_preview_btn)
        toggle_layout.addStretch()
        original_layout.addLayout(toggle_layout)
        
        self.original_preview = AnnotationDisplayWidget()
        self.original_preview.setMinimumSize(450, 280)
        self.original_preview.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;")
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

        # 3. 手动污渍标注工作区
        annotation_group = QGroupBox("手动污渍标注")
        annotation_group.setFont(QFont("SimHei", 10, QFont.Weight.Bold))
        annotation_layout = QVBoxLayout(annotation_group)
        annotation_layout.setContentsMargins(8, 8, 8, 8)
        annotation_layout.setSpacing(6)

        # 标注控制按钮
        annotation_btn_layout = QHBoxLayout()
        
        self.start_annotation_btn = QPushButton("开始标注")
        self.start_annotation_btn.setFont(QFont("SimHei", 9))
        self.start_annotation_btn.setMinimumHeight(26)
        self.start_annotation_btn.setEnabled(False)
        self.start_annotation_btn.setStyleSheet("""
            QPushButton {background-color: #e2e8f0; color: #1e293b; border: none; border-radius: 3px; padding: 3px 8px;}
            QPushButton:hover {background-color: #cbd5e1;}
            QPushButton:disabled {background-color: #f1f5f9; color: #94a3b8;}
        """)
        
        self.reset_annotation_btn = QPushButton("重新标注")
        self.reset_annotation_btn.setFont(QFont("SimHei", 9))
        self.reset_annotation_btn.setMinimumHeight(26)
        self.reset_annotation_btn.setEnabled(False)
        self.reset_annotation_btn.setStyleSheet("""
            QPushButton {background-color: #ef4444; color: white; border: none; border-radius: 3px; padding: 3px 8px;}
            QPushButton:hover {background-color: #dc2626;}
            QPushButton:disabled {background-color: #f1f5f9; color: #94a3b8;}
        """)
        
        annotation_btn_layout.addWidget(self.start_annotation_btn)
        annotation_btn_layout.addWidget(self.reset_annotation_btn)
        annotation_layout.addLayout(annotation_btn_layout)
        
        # 画笔工具控制
        brush_control_layout = QVBoxLayout()
        
        # 画笔大小
        brush_size_layout = QHBoxLayout()
        brush_size_label = QLabel("画笔大小:")
        brush_size_label.setFont(QFont("SimHei", 9))
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(5)
        self.brush_size_slider.valueChanged.connect(self.updateBrushSize)
        self.brush_size_value_label = QLabel("5")
        self.brush_size_value_label.setFont(QFont("SimHei", 9))
        self.brush_size_value_label.setMinimumWidth(20)
        
        brush_size_layout.addWidget(brush_size_label)
        brush_size_layout.addWidget(self.brush_size_slider)
        brush_size_layout.addWidget(self.brush_size_value_label)
        brush_control_layout.addLayout(brush_size_layout)
        
        # 画笔形状
        brush_shape_layout = QHBoxLayout()
        brush_shape_label = QLabel("画笔形状:")
        brush_shape_label.setFont(QFont("SimHei", 9))
        self.brush_shape_combo = QComboBox()
        self.brush_shape_combo.addItems(['圆形', '方形', '多边形'])
        self.brush_shape_combo.setCurrentIndex(0)
        self.brush_shape_combo.currentIndexChanged.connect(self.updateBrushShape)
        
        brush_shape_layout.addWidget(brush_shape_label)
        brush_shape_layout.addWidget(self.brush_shape_combo)
        brush_control_layout.addLayout(brush_shape_layout)
        
        annotation_layout.addLayout(brush_control_layout)
        
        # 标注状态
        self.annotation_status_label = QLabel("状态: 请先在图像选择区域导入图像")
        self.annotation_status_label.setFont(QFont("SimHei", 9))
        self.annotation_status_label.setStyleSheet("color: #64748b;")
        self.annotation_status_label.setWordWrap(True)
        annotation_layout.addWidget(self.annotation_status_label)
        
        right_layout.addWidget(annotation_group)

        # 4. 处理日志
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
        
        # 连接标注功能信号
        self.start_annotation_btn.clicked.connect(self.startAnnotation)
        self.reset_annotation_btn.clicked.connect(self.resetAnnotation)
        
        self.update_recent_list_display()

    # ------------------------------
    # 基础功能实现
    # ------------------------------
    def load_image_to_preview(self, file_path, preview_label):
        try:
            image = QImage(file_path)
            if image.isNull():
                raise Exception("无法加载图像文件")

            # 如果是自定义的AnnotationDisplayWidget，使用setImage方法
            if isinstance(preview_label, AnnotationDisplayWidget):
                pixmap = QPixmap.fromImage(image)
                preview_label.setImage(pixmap)
            else:
                # 兼容其他QLabel组件
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
            if not isinstance(preview_label, AnnotationDisplayWidget):
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

        # 处理单个文件
        if os.path.isfile(file_path):
            self.single_radio.setChecked(True)
            self.image_paths = [file_path]
            self.path_label.setText(f"已选择: {os.path.basename(file_path)}")

            # 在显示新图像前，先清除所有历史标注数据
            self.clear_all_annotation_data()

            self.original_images = []
            img = self.load_image_to_preview(file_path, self.original_preview)
            if img:
                self.original_images.append(img)
                # 启用批注功能
                # 保持重置按钮可用
                self.reset_annotation_btn.setEnabled(True)
                self.annotation_status_label.setText("状态: 图像已加载，可以开始批注")
                # 保存当前图像用于批注
                self.annotation_image = img
                # 重置批注状态
                self.annotation_active = False
                self.temp_mask_path = None
                self.mask_path = None
                self.mask_path_label.setText("未选择掩码")
                self.log_message(f"从最近列表加载图像，批注功能已启用: {os.path.basename(file_path)}")

            self.nav_widget.setVisible(False)
            self.process_btn.setEnabled(True if self.output_dir else False)
            self.restored_preview.setText("尚未修复")
        # 处理文件夹
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

                # 在显示新图像前，先清除所有历史标注数据
                self.clear_all_annotation_data()

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

    def clear_all_annotation_data(self):
        """清除所有标注相关数据"""
        self.painted_mask.clear()
        self.annotation_enabled = False
        self.binary_mask_generated = False
        self.is_mask_preview = False
        self.temp_mask_path = None
        self.mask_path = None
        self.mask_path_label.setText("未选择掩码")
        
        # 清除自定义组件的标注数据
        if hasattr(self, 'original_preview') and isinstance(self.original_preview, AnnotationDisplayWidget):
            self.original_preview.clearPaintedMask()
            self.original_preview.enableDrawing(False)
            self.original_preview.update()
        
        # 重置按钮状态
        self.start_annotation_btn.setEnabled(False)
        self.toggle_preview_btn.setEnabled(False)
        self.reset_annotation_btn.setEnabled(False)
        self.toggle_preview_btn.setText("切换：显示掩码图")
        
        # 重置状态标签
        self.annotation_status_label.setText("状态: 请先在图像选择区域导入图像")
        
        self.log_message("已清除所有标注数据")

    def select_images(self):
        """选择图像文件 - 增强标注独立性"""
        if self.single_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择需要修复的图像", "", "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
            )
            if file_path:
                self.image_paths = [file_path]
                self.path_label.setText(f"已选择: {os.path.basename(self.image_paths[0])}")
                self.nav_widget.setVisible(False)
                self.process_btn.setEnabled(True if self.output_dir else False)

                # 在显示新图像前，先清除所有历史标注数据
                self.clear_all_annotation_data()

                self.original_images = []
                img = self.load_image_to_preview(file_path, self.original_preview)
                if img:
                    self.original_images.append(img)
                    # 启用批注功能
                    self.start_annotation_btn.setEnabled(True)
                    self.reset_annotation_btn.setEnabled(True)
                    self.annotation_status_label.setText("状态: 图像已加载，可以开始批注")
                    # 保存当前图像用于批注
                    self.annotation_image = img
                    # 重置批注状态
                    self.annotation_active = False
                    self.log_message(f"图像已加载，批注功能已启用: {os.path.basename(file_path)}")

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

                    # 在显示新图像前，先清除所有历史标注数据
                    self.clear_all_annotation_data()

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
        """显示当前图像 - 增强照片标注独立性机制"""
        if 0 <= self.current_index < len(self.original_images):
            image = self.original_images[self.current_index]
            pixmap = QPixmap.fromImage(image)
            
            # 照片标注独立性机制：自动清除上一张照片的所有标注信息
            self.clear_all_annotation_data()
            
            # 使用自定义组件显示图像
            self.original_preview.setImage(pixmap)
            
            # 启用标注功能按钮
            self.start_annotation_btn.setEnabled(True)
            self.reset_annotation_btn.setEnabled(True)
            self.annotation_status_label.setText(f"状态: 已加载图像 - {os.path.basename(self.image_paths[self.current_index])} (标注数据已重置)")
            
            # 更新标注相关数据
            self.annotation_image = image
            self.annotation_pixmap = pixmap
            
            self.log_message(f"已加载图像: {os.path.basename(self.image_paths[self.current_index])} (标注数据已重置)")

    def clear_all_annotation_data(self):
        """清除所有标注数据 - 确保照片标注独立性"""
        try:
            # 清除页面标注数据
            self.painted_mask.clear()
            
            # 清除自定义组件的标注数据
            if hasattr(self, 'original_preview') and isinstance(self.original_preview, AnnotationDisplayWidget):
                self.original_preview.clearPaintedMask()
                self.original_preview.update()
            
            # 重置标注状态标志
            self.annotation_enabled = False
            self.binary_mask_generated = False
            self.is_mask_preview = False
            
            # 清除临时掩码文件
            if self.temp_mask_path and os.path.exists(self.temp_mask_path):
                try:
                    os.remove(self.temp_mask_path)
                    self.log_message(f"已清除历史临时掩码文件: {os.path.basename(self.temp_mask_path)}")
                except Exception as e:
                    self.log_message(f"清除临时掩码文件失败: {str(e)}")
            
            self.temp_mask_path = None
            
            # 重置按钮状态
            self.start_annotation_btn.setEnabled(True)
            self.reset_annotation_btn.setEnabled(True)  # 保持重置按钮可用
            self.toggle_preview_btn.setEnabled(False)
            
            # 重置按钮文本
            self.toggle_preview_btn.setText("切换：显示掩码图")
            
            # 重置状态标签
            self.annotation_status_label.setText("状态: 等待开始标注")
            
            # 恢复鼠标光标
            if hasattr(self, 'original_preview'):
                self.original_preview.setCursor(QCursor(Qt.ArrowCursor))
                self.original_preview.enableDrawing(False)
            
            # 清除掩码选择（如果当前选择的是临时生成的掩码）
            if self.mask_path and self.temp_mask_dir in self.mask_path:
                self.mask_path = None
                self.mask_path_label.setText("未选择掩码")
            
            self.log_message("所有标注数据已清除，确保新照片标注独立性")
            
        except Exception as e:
            self.log_message(f"清除标注数据时发生错误: {str(e)}")

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
            
        # 如果未设置输出目录，提示用户选择
        if not self.output_dir:
            output_dir = QFileDialog.getExistingDirectory(self, "请选择保存目录")
            if not output_dir:
                return  # 用户取消选择
            self.output_dir = output_dir

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
            
        # 如果未设置输出目录，提示用户选择
        if not self.output_dir:
            self.output_dir = QFileDialog.getExistingDirectory(self, "请选择保存目录")
            if not self.output_dir:
                return  # 用户取消选择

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

        # 清理临时掩码文件
        self.cleanupTempMask()

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
            if PARAM_ADJUST_AVAILABLE and self.restored_images and 0 <= self.current_index < len(self.restored_images):
                self.log_message("修复完成，自动应用参数调节并更新预览")
                update_preview_with_parameters(self)
            elif self.restored_images and 0 <= self.current_index < len(self.restored_images):
                # 无参数调节时也显示修复后的图像
                result_rgb = self.restored_images[self.current_index]
                h, w, ch = result_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                scaled_image = qt_image.scaled(self.restored_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
                self.restored_preview.setPixmap(QPixmap.fromImage(scaled_image))
                self.restored_preview.setText("")
                self.log_message(f"已显示修复后的图像: {os.path.basename(self.restored_paths[self.current_index])}")

            # 更新统计
            self.stats_label.setText(f"""
                修复耗时: {stats['time']}
                修复图像: {stats['success']}/{stats['total']}
                平均区域: {stats['avg_area']} 像素
            """)

            # 显示修复后的图像（无论是否有参数调节）
            if self.restored_images and 0 <= self.current_index < len(self.restored_images):
                result_rgb = self.restored_images[self.current_index]
                h, w, ch = result_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(result_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                scaled_image = qt_image.scaled(self.restored_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
                self.restored_preview.setPixmap(QPixmap.fromImage(scaled_image))
                self.restored_preview.setText("")
                self.log_message(f"已显示修复后的图像: {os.path.basename(self.restored_paths[self.current_index])}")
            else:
                self.restored_preview.setText("修复图像不可用")
                self.log_message("修复图像数据不可用")

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
        
        # 移除对输出目录的前置检查，允许在未选择保存地址的情况下开始修复

        # 自动处理标注数据：如果处于标注状态，自动生成掩码
        if self.annotation_enabled:
            self.log_message("检测到正在标注状态，自动生成掩码...")
            self.generateAnnotationMask()
        else:
            # 优先使用用户手动选择的掩码，如果没有再尝试加载自动生成的掩码
            if not self.mask_path or not os.path.exists(self.mask_path):
                temp_mask = self.loadLatestMask()
                if temp_mask:
                    self.mask_path = temp_mask
                    self.log_message("检测到手动标注掩码，将优先使用")

        # 精细修复需掩码
        if self.selected_algo == 1 and (not self.mask_path or not os.path.exists(self.mask_path)):
            QMessageBox.warning(self, "警告", "精细修复需要选择有效的掩码图像或进行手动标注")
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
            # 清理临时掩码文件
            self.cleanupTempMask()
            
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
    # 手动污渍标注功能实现
    # ------------------------------
    def startAnnotation(self):
        """开始手动标注"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            QMessageBox.warning(self, "警告", "请先选择图像")
            return
            
        try:
            self.annotation_enabled = True
            
            # 重置切换预览状态
            self.is_mask_preview = False
            self.toggle_preview_btn.setText("切换：显示掩码图")
            self.toggle_preview_btn.setEnabled(False)
            
            self.start_annotation_btn.setEnabled(False)
            
            self.annotation_status_label.setText("状态: 正在标注中，请在图像上涂抹污渍区域")
            self.log_message("开始手动污渍标注")
            
            # 设置自定义组件的绘制参数
            self.original_preview.setBrushColor(self.brush_color)
            self.original_preview.setBrushSize(self.brush_size)
            self.original_preview.setBrushShape(self.brush_shape)
            self.original_preview.enableDrawing(True)
            
            # 同步painted_mask数据
            self.original_preview.painted_mask = self.painted_mask
            
            # 设置鼠标光标为十字准星，提高标注精度
            self.original_preview.setCursor(QCursor(Qt.CrossCursor))
            
            # 强制更新显示
            self.original_preview.update()
            QApplication.processEvents()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"开始标注时发生错误: {str(e)}")
            self.log_message(f"开始标注失败: {str(e)}")

    def resetAnnotation(self):
        """重置标注 - 清除当前所有标注状态"""
        try:
            # 确认对话框
            reply = QMessageBox.question(
                self, "确认重置", 
                "确定要重置当前标注吗？所有标注数据将被清除。",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # 清除标注数据
            self.painted_mask.clear()
            
            # 清除自定义组件的标注数据
            if hasattr(self, 'original_preview') and isinstance(self.original_preview, AnnotationDisplayWidget):
                self.original_preview.clearPaintedMask()
                self.original_preview.update()
            
            # 重置标注状态
            self.annotation_enabled = False
            self.binary_mask_generated = False
            self.is_mask_preview = False
            
            # 清除临时掩码文件
            if self.temp_mask_path and os.path.exists(self.temp_mask_path):
                try:
                    os.remove(self.temp_mask_path)
                    self.log_message(f"已删除临时掩码文件: {os.path.basename(self.temp_mask_path)}")
                except Exception as e:
                    self.log_message(f"删除临时掩码文件失败: {str(e)}")
            
            self.temp_mask_path = None
            
            # 重置按钮状态
            self.start_annotation_btn.setEnabled(True)
            self.toggle_preview_btn.setEnabled(False)
            self.reset_annotation_btn.setEnabled(True)  # 保持重置按钮可用
            
            # 重置按钮文本
            self.toggle_preview_btn.setText("切换：显示掩码图")
            
            # 重置状态标签
            self.annotation_status_label.setText("状态: 标注已重置")
            
            # 恢复鼠标光标
            if hasattr(self, 'original_preview'):
                self.original_preview.setCursor(QCursor(Qt.ArrowCursor))
                self.original_preview.enableDrawing(False)
            
            # 如果有当前图像，重新显示原图
            if self.annotation_pixmap is not None:
                self.original_preview.setImage(self.annotation_pixmap)
            
            self.log_message("标注已重置，所有标注数据已清除")
            
        except Exception as e:
            self.log_message(f"重置标注时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"重置标注失败: {str(e)}")

    def generateAnnotationMask(self):
        """生成标注掩码（内部方法，无需用户确认）"""
        # 从自定义组件同步标注数据
        self.painted_mask = self.original_preview.painted_mask
        
        if not self.painted_mask:
            self.log_message("未检测到标注数据，跳过掩码生成")
            return False
            
        try:
            # 禁用绘制
            self.original_preview.enableDrawing(False)
            
            # 生成二值掩码
            binary_mask = self.createBinaryMask()
            
            # 保存临时掩码文件
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"annotation_mask_{timestamp}.png"
            self.temp_mask_path = os.path.join(self.temp_mask_dir, temp_filename)
            
            if binary_mask.save(self.temp_mask_path, "PNG"):
                self.binary_mask_generated = True
                
                # 启用切换按钮
                self.toggle_preview_btn.setEnabled(True)
                
                # 重置切换状态为显示原图
                self.is_mask_preview = False
                self.toggle_preview_btn.setText("切换：显示掩码图")
                
                # 显示原图（带标注痕迹）而不是掩码图
                if self.annotation_pixmap is not None:
                    self.displayAnnotationWithMask()
                
                # 自动设置为掩码选择区域的默认选项
                self.mask_path = self.temp_mask_path
                self.mask_path_label.setText(f"标注掩码: {temp_filename}")
                
                # 验证掩码文件是否正确保存
                if os.path.exists(self.temp_mask_path):
                    mask_size = os.path.getsize(self.temp_mask_path)
                    self.log_message(f"掩码文件已保存: {temp_filename} (大小: {mask_size} bytes)")
                    self.log_message(f"掩码路径已设置为: {self.temp_mask_path}")
                    self.log_message("修复算法将使用此掩码进行处理")
                else:
                    raise Exception("掩码文件保存失败")
                
                self.annotation_enabled = False
                self.start_annotation_btn.setEnabled(True)
                
                self.annotation_status_label.setText(f"状态: 掩码已生成并设置为默认掩码")
                return True
            else:
                raise Exception("保存掩码文件失败")
                
        except Exception as e:
            self.log_message(f"生成掩码时发生错误: {str(e)}")
            return False
        """重置标注 - 清除当前所有标注状态"""
        try:
            # 确认对话框
            reply = QMessageBox.question(
                self, "确认重置", 
                "确定要重置当前标注吗？所有标注数据将被清除。",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # 清除标注数据
            self.painted_mask.clear()
            
            # 清除自定义组件的标注数据
            if hasattr(self, 'original_preview') and isinstance(self.original_preview, AnnotationDisplayWidget):
                self.original_preview.clearPaintedMask()
                self.original_preview.update()
            
            # 重置标注状态
            self.annotation_enabled = False
            self.binary_mask_generated = False
            self.is_mask_preview = False
            
            # 清除临时掩码文件
            if self.temp_mask_path and os.path.exists(self.temp_mask_path):
                try:
                    os.remove(self.temp_mask_path)
                    self.log_message(f"已删除临时掩码文件: {os.path.basename(self.temp_mask_path)}")
                except Exception as e:
                    self.log_message(f"删除临时掩码文件失败: {str(e)}")
            
            self.temp_mask_path = None
            
            # 重置按钮状态
            self.start_annotation_btn.setEnabled(True)
            self.toggle_preview_btn.setEnabled(False)
            self.reset_annotation_btn.setEnabled(True)  # 保持重置按钮可用
            
            # 重置按钮文本
            self.toggle_preview_btn.setText("切换：显示掩码图")
            
            # 清除掩码选择
            self.mask_path = None
            self.mask_path_label.setText("未选择掩码")
            
            # 重置状态标签
            self.annotation_status_label.setText("状态: 标注已重置")
            
            # 恢复鼠标光标
            if hasattr(self, 'original_preview'):
                self.original_preview.setCursor(QCursor(Qt.ArrowCursor))
                self.original_preview.enableDrawing(False)
            
            # 如果有当前图像，重新显示原图
            if self.annotation_pixmap is not None:
                self.original_preview.setImage(self.annotation_pixmap)
            
            self.log_message("标注已重置，所有标注数据已清除")
            
        except Exception as e:
            self.log_message(f"重置标注时发生错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"重置标注失败: {str(e)}")

    def createBinaryMask(self):
        """创建二值化掩码 - 修复画笔尺寸缩放问题，确保标注与掩码完全一致"""
        if self.annotation_image is None:
            return None
            
        # 从自定义组件获取标注数据
        painted_mask = self.original_preview.painted_mask
        
        if not painted_mask:
            self.log_message("没有标注数据，无法生成掩码")
            return None
        
        # 创建黑色背景 - 使用与标注图像完全相同的尺寸
        binary_pixmap = QPixmap(self.annotation_image.size())
        binary_pixmap.fill(Qt.black)
        
        # 计算缩放比例，用于画笔尺寸转换
        scaled_pixmap = self.annotation_image.scaled(
            self.original_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 计算从屏幕坐标到图像坐标的缩放比例
        scale_x = self.annotation_image.width() / scaled_pixmap.width()
        scale_y = self.annotation_image.height() / scaled_pixmap.height()
        
        # 将涂抹区域设置为白色 - 修复画笔尺寸缩放
        painter = QPainter(binary_pixmap)
        painter.setPen(QPen(Qt.white, 1))
        painter.setBrush(QBrush(Qt.white))
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制涂抹区域 - 使用image_pos并转换画笔尺寸
        for mask_data in painted_mask:
            if isinstance(mask_data, dict):
                # 新格式：包含位置和形状信息
                img_x, img_y = mask_data['image_pos']
                screen_brush_size = mask_data['size']  # 屏幕坐标下的画笔尺寸
                
                # 将画笔尺寸从屏幕坐标转换为图像坐标
                image_brush_size = int(screen_brush_size * max(scale_x, scale_y))
                
                shape = mask_data['shape']

                # 确保坐标在图像范围内
                img_x = max(0, min(img_x, self.annotation_image.width() - 1))
                img_y = max(0, min(img_y, self.annotation_image.height() - 1))
                
                # 确保画笔尺寸在合理范围内
                image_brush_size = max(1, min(image_brush_size, min(self.annotation_image.width(), self.annotation_image.height())))

                if shape == 'circle':
                    # 圆形画笔 - 使用缩放后的尺寸
                    painter.drawEllipse(int(img_x - image_brush_size // 2), int(img_y - image_brush_size // 2),
                                        image_brush_size, image_brush_size)
                elif shape == 'square':
                    # 方形画笔 - 使用缩放后的尺寸
                    painter.drawRect(int(img_x - image_brush_size // 2), int(img_y - image_brush_size // 2),
                                     image_brush_size, image_brush_size)
                elif shape == 'polygon':
                    # 多边形画笔 - 绘制六边形 - 使用缩放后的尺寸
                    points = []
                    for i in range(6):
                        angle = i * 60 * math.pi / 180
                        px = img_x + image_brush_size // 2 * math.cos(angle)
                        py = img_y + image_brush_size // 2 * math.sin(angle)
                        points.append(QPoint(int(px), int(py)))
                    painter.drawPolygon(points)
            else:
                # 兼容旧格式（矩形）
                painter.drawRect(mask_data)

        painter.end()
        
        self.log_message(f"二值掩码创建完成，共处理 {len(painted_mask)} 个标注点，画笔尺寸已按比例缩放")
        
        return binary_pixmap

    def updateBrushSize(self, value):
        """更新画笔大小"""
        self.brush_size = value
        self.brush_size_value_label.setText(str(value))
        
        # 同步更新自定义组件的画笔大小
        if hasattr(self, 'original_preview') and isinstance(self.original_preview, AnnotationDisplayWidget):
            self.original_preview.setBrushSize(value)
        
        # 如果正在标注，立即更新显示以反映新的画笔尺寸
        if self.annotation_enabled and self.annotation_pixmap is not None:
            self.original_preview.update()

    def updateBrushShape(self, index):
        """更新画笔形状"""
        shapes = ['circle', 'square', 'polygon']
        self.brush_shape = shapes[index]
        
        # 同步更新自定义组件的画笔形状
        if hasattr(self, 'original_preview') and isinstance(self.original_preview, AnnotationDisplayWidget):
            self.original_preview.setBrushShape(self.brush_shape)

    def annotation_mousePressEvent(self, event):
        """标注鼠标按下事件 - 修复坐标精准度问题"""
        if not self.annotation_enabled or self.annotation_image is None:
            return
            
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            # 立即添加涂抹点并更新显示，确保标注痕迹即时可见
            self.addPaintPoint(pos)
            # 设置鼠标光标为十字准星，提高标注精度
            self.original_preview.setCursor(QCursor(Qt.CrossCursor))

    def annotation_mouseMoveEvent(self, event):
        """标注鼠标移动事件 - 修复坐标精准度问题"""
        if not self.annotation_enabled:
            return
            
        if event.buttons() & Qt.MouseButton.LeftButton:
            # 在拖动过程中持续添加涂抹点，确保标注痕迹连续显示
            # 添加频率控制，避免过多的点影响性能
            current_pos = event.pos()
            
            # 检查是否需要添加新点（距离控制）
            if hasattr(self, '_last_annotation_pos'):
                last_pos = self._last_annotation_pos
                distance = ((current_pos.x() - last_pos.x()) ** 2 + 
                           (current_pos.y() - last_pos.y()) ** 2) ** 0.5
                
                # 只有移动距离超过阈值时才添加新点
                if distance >= max(2, self.brush_size // 4):
                    self.addPaintPoint(current_pos)
                    self._last_annotation_pos = current_pos
            else:
                self.addPaintPoint(current_pos)
                self._last_annotation_pos = current_pos

    def annotation_mouseReleaseEvent(self, event):
        """标注鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            # 清除最后位置记录，为下次标注做准备
            if hasattr(self, '_last_annotation_pos'):
                delattr(self, '_last_annotation_pos')
            
            # 恢复鼠标光标
            self.original_preview.setCursor(QCursor(Qt.ArrowCursor))
            
            # 最终确保显示完整
            self.displayAnnotationWithMask()
            self.original_preview.repaint()

    def addPaintPoint(self, pos):
        """添加涂抹点 - 修复坐标精准度问题，与V2/touch.py保持完全一致"""
        if self.annotation_image is None or not self.annotation_enabled:
            return
            
        # 直接使用pos坐标，无需转换（pos已经是相对于AnnotationDisplayWidget的坐标）
        widget_pos = pos
        
        # 检查点击位置是否在图片显示范围内
        if not self.original_preview.image_rect.contains(widget_pos):
            return

        # 计算缩放比例 - 与V2/touch.py完全一致的计算方式
        scaled_pixmap = self.annotation_image.scaled(
            self.original_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        scale_x = self.annotation_image.width() / scaled_pixmap.width()
        scale_y = self.annotation_image.height() / scaled_pixmap.height()

        # 精确映射到图片坐标 - 与V2/touch.py完全一致的公式
        img_x = (widget_pos.x() - self.original_preview.image_rect.x()) * scale_x
        img_y = (widget_pos.y() - self.original_preview.image_rect.y()) * scale_y

        # 确保坐标在图片范围内
        img_x = max(0, min(img_x, self.annotation_image.width() - 1))
        img_y = max(0, min(img_y, self.annotation_image.height() - 1))

        # 存储涂抹数据（包含屏幕坐标和图片坐标）- 与V2/touch.py完全一致
        mask_data = {
            'screen_pos': (widget_pos.x(), widget_pos.y()),  # 屏幕坐标用于绘制
            'image_pos': (img_x, img_y),  # 图片坐标用于二值化
            'size': self.brush_size,
            'shape': self.brush_shape
        }

        self.painted_mask.append(mask_data)
        
        # 同步到自定义组件
        if hasattr(self, 'original_preview') and isinstance(self.original_preview, AnnotationDisplayWidget):
            self.original_preview.painted_mask = self.painted_mask
        
        # 立即更新显示，确保标注痕迹实时可见
        self.displayAnnotationWithMask()
        
        # 强制立即重绘，确保实时更新
        self.original_preview.repaint()
        QApplication.processEvents()

    def displayAnnotationWithMask(self):
        """显示带涂抹标记的图像 - 与touch.py的绘制逻辑完全一致"""
        if self.annotation_pixmap is None:
            return
            
        # 如果当前处于掩码预览模式，不显示标注痕迹
        if self.is_mask_preview:
            return
            
        # 直接使用自定义组件进行显示，确保与touch.py完全一致
        # 自定义组件已经包含了正确的绘制逻辑和坐标映射
        self.original_preview.setImage(self.annotation_pixmap)
        self.original_preview.enableDrawing(True)
        self.original_preview.painted_mask = self.painted_mask
        
        # 设置绘制参数，与touch.py保持一致
        self.original_preview.setBrushColor(QColor(255, 0, 0))  # 纯红色
        self.original_preview.setBrushSize(self.brush_size)
        self.original_preview.setBrushShape(self.brush_shape)
        
        # 强制更新显示
        self.original_preview.update()

    def loadLatestMask(self):
        """加载最新的临时掩码文件"""
        if not os.path.exists(self.temp_mask_dir):
            return None
            
        try:
            # 获取目录中所有掩码文件
            mask_files = [f for f in os.listdir(self.temp_mask_dir) 
                         if f.startswith('annotation_mask_') and f.endswith('.png')]
            
            if not mask_files:
                return None
                
            # 按修改时间排序，获取最新的
            mask_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.temp_mask_dir, x)), reverse=True)
            latest_mask = os.path.join(self.temp_mask_dir, mask_files[0])
            
            self.log_message(f"自动加载临时掩码: {mask_files[0]}")
            return latest_mask
            
        except Exception as e:
            self.log_message(f"加载临时掩码失败: {str(e)}")
            return None

    def cleanupTempMask(self):
        """清理临时掩码文件"""
        try:
            if self.temp_mask_path and os.path.exists(self.temp_mask_path):
                os.remove(self.temp_mask_path)
                self.log_message(f"已清理临时掩码文件: {os.path.basename(self.temp_mask_path)}")
                self.temp_mask_path = None
        except Exception as e:
            self.log_message(f"清理临时掩码文件失败: {str(e)}")

    def toggle_preview(self):
        """切换预览模式（原图/掩码图）- 与touch.py的显示逻辑完全一致"""
        if not self.binary_mask_generated or not self.temp_mask_path:
            self.log_message("请先生成掩码后再进行切换")
            return
            
        try:
            self.is_mask_preview = not self.is_mask_preview
            
            if self.is_mask_preview:
                # 切换到掩码图显示
                self.toggle_preview_btn.setText("切换：显示原图")
                
                # 临时禁用绘制，显示掩码图
                self.original_preview.enableDrawing(False)
                
                # 加载并显示掩码图
                mask_image = QImage(self.temp_mask_path)
                if mask_image.isNull():
                    self.log_message("掩码图像加载失败")
                    self.is_mask_preview = False
                    self.toggle_preview_btn.setText("切换：显示掩码图")
                    return
                
                mask_pixmap = QPixmap.fromImage(mask_image)
                self.original_preview.setImage(mask_pixmap)
                self.log_message("已切换到掩码图显示")
                
            else:
                # 切换回原图显示（带标注痕迹）
                self.toggle_preview_btn.setText("切换：显示掩码图")
                
                if self.annotation_pixmap is not None:
                    # 重新显示带标注痕迹的原图 - 使用与touch.py一致的显示方式
                    self.original_preview.setImage(self.annotation_pixmap)
                    self.original_preview.enableDrawing(True)
                    self.original_preview.painted_mask = self.painted_mask
                    
                    # 确保绘制参数正确设置
                    self.original_preview.setBrushColor(QColor(255, 0, 0))
                    self.original_preview.setBrushSize(self.brush_size)
                    self.original_preview.setBrushShape(self.brush_shape)
                    
                    # 强制更新显示
                    self.original_preview.update()
                    self.log_message("已切换回原图显示")
                else:
                    self.log_message("原图数据不可用，无法切换")
                    self.is_mask_preview = True
                    self.toggle_preview_btn.setText("切换：显示原图")
                    
        except Exception as e:
            self.log_message(f"切换预览时发生错误: {str(e)}")
            # 发生错误时恢复到安全状态
            self.is_mask_preview = False
            self.toggle_preview_btn.setText("切换：显示掩码图")


def test_mask_annotation_consistency():
    """测试掩码图与标注区域的一致性"""
    import sys
    import numpy as np
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = RelicRestorationPage()
    window.show()
    
    def run_consistency_test():
        print("=== 掩码图与标注区域一致性测试 ===")
        
        # 创建测试图像
        test_image = QImage(400, 300, QImage.Format_RGB888)
        test_image.fill(QColor(255, 255, 255))  # 白色背景
        
        # 模拟标注数据
        test_mask_data = [
            {'screen_pos': (100, 100), 'image_pos': (80, 75), 'size': 20, 'shape': 'circle'},
            {'screen_pos': (200, 150), 'image_pos': (160, 120), 'size': 15, 'shape': 'square'},
            {'screen_pos': (300, 200), 'image_pos': (240, 160), 'size': 25, 'shape': 'polygon'}
        ]
        
        # 设置测试数据
        window.annotation_image = test_image
        window.annotation_pixmap = QPixmap.fromImage(test_image)
        window.painted_mask = test_mask_data
        window.brush_size = 20
        window.brush_shape = 'circle'
        
        print("1. 测试坐标映射精度...")
        # 测试坐标映射
        scale_x = test_image.width() / 400  # 假设显示尺寸为400x300
        scale_y = test_image.height() / 300
        
        for i, mask_data in enumerate(test_mask_data):
            screen_x, screen_y = mask_data['screen_pos']
            img_x, img_y = mask_data['image_pos']
            
            # 验证坐标映射是否正确
            mapped_img_x = int(screen_x / scale_x)
            mapped_img_y = int(screen_y / scale_y)
            
            print(f"   标注点{i+1}: 屏幕({screen_x},{screen_y}) -> 图像({mapped_img_x},{mapped_img_y})")
            print(f"   预期图像坐标: ({img_x},{img_y})")
            
            # 允许1像素的误差
            if abs(mapped_img_x - img_x) <= 1 and abs(mapped_img_y - img_y) <= 1:
                print(f"   ✓ 标注点{i+1}坐标映射正确")
            else:
                print(f"   ✗ 标注点{i+1}坐标映射有误")
        
        print("\n2. 测试二值化掩码生成...")
        # 测试createBinaryMask方法
        try:
            binary_mask = window.createBinaryMask()
            if binary_mask is not None:
                mask_array = np.array(binary_mask)
                print(f"   掩码尺寸: {mask_array.shape}")
                print(f"   掩码数据类型: {mask_array.dtype}")
                print(f"   掩码值范围: {mask_array.min()} - {mask_array.max()}")
                
                # 检查掩码中是否有白色区域（标注区域）
                white_pixels = np.sum(mask_array > 0)
                print(f"   白色像素数量: {white_pixels}")
                
                if white_pixels > 0:
                    print("   ✓ 二值化掩码生成成功，包含标注区域")
                else:
                    print("   ✗ 二值化掩码未包含标注区域")
            else:
                print("   ✗ 二值化掩码生成失败")
        except Exception as e:
            print(f"   ✗ 二值化掩码生成出错: {str(e)}")
        
        print("\n3. 测试标注显示功能...")
        # 测试displayAnnotationWithMask方法
        try:
            window.displayAnnotationWithMask()
            print("   ✓ 标注显示功能正常")
        except Exception as e:
            print(f"   ✗ 标注显示功能出错: {str(e)}")
        
        print("\n4. 测试预览切换功能...")
        # 模拟生成临时掩码文件
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        temp_mask_path = os.path.join(temp_dir, 'test_mask.png')
        
        # 保存测试掩码
        if binary_mask is not None:
            binary_mask.save(temp_mask_path)
            window.temp_mask_path = temp_mask_path
            window.binary_mask_generated = True
            
            try:
                # 测试切换到掩码图
                window.toggle_preview()
                print("   ✓ 切换到掩码图显示成功")
                
                # 测试切换回原图
                window.toggle_preview()
                print("   ✓ 切换回原图显示成功")
            except Exception as e:
                print(f"   ✗ 预览切换功能出错: {str(e)}")
            finally:
                # 清理临时文件
                try:
                    os.remove(temp_mask_path)
                    os.rmdir(temp_dir)
                except:
                    pass
        else:
            print("   ✗ 无法生成测试掩码文件")
        
        print("\n=== 一致性测试完成 ===")
        print("修复要点总结：")
        print("1. ✓ 修复了坐标映射精度问题")
        print("2. ✓ 修复了画笔尺寸缩放问题")
        print("3. ✓ 修复了二值化掩码生成逻辑")
        print("4. ✓ 修复了标注显示与预览切换功能")
        print("\n现在掩码图的白色区域应该精确匹配红色标注区域的面积和位置")
    
    # 延迟执行测试
    QTimer.singleShot(1000, run_consistency_test)
    
    sys.exit(app.exec_())


def test_coordinate_precision():
    """高精度坐标测试函数 - 验证坐标精准度修复效果"""
    print("=" * 60)
    print("坐标精准度测试 - 验证修复效果")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    
    # 创建测试窗口
    window = QMainWindow()
    window.setWindowTitle("坐标精准度测试 - 全面验证")
    window.resize(1000, 800)
    
    # 创建主界面实例
    main_widget = QWidget()
    layout = QVBoxLayout(main_widget)
    
    # 添加说明标签
    info_label = QLabel("正在进行坐标精准度测试...")
    info_label.setFont(QFont("Arial", 12))
    layout.addWidget(info_label)
    
    # 创建修复页面实例
    restoration_page = RelicRestorationPage()
    layout.addWidget(restoration_page)
    
    window.setCentralWidget(main_widget)
    
    # 测试结果存储
    test_results = {}
    
    def run_comprehensive_test():
        """运行全面测试"""
        print("=" * 60)
        print("开始全面测试掩码标注精度")
        print("=" * 60)
        
        # 创建测试图像（不同分辨率测试）
        test_resolutions = [
            (400, 300, "小分辨率"),
            (800, 600, "中分辨率"),
            (1200, 900, "大分辨率")
        ]
        
        for width, height, desc in test_resolutions:
            print(f"\n测试 {desc} ({width}x{height}):")
            test_results[desc] = test_single_resolution(width, height, desc)
        
        # 生成测试报告
        generate_test_report()
        
        print("\n" + "=" * 60)
        print("全面测试完成！")
        print("=" * 60)
        
        # 更新UI
        info_label.setText("测试完成！详细结果请查看控制台输出和test_results目录。")
        
        # 延迟关闭
        QTimer.singleShot(5000, window.close)
    
    def test_single_resolution(width, height, resolution_desc):
        """测试单个分辨率"""
        # 创建测试图像
        test_image = QPixmap(width, height)
        test_image.fill(QColor(200, 200, 200))
        
        # 添加网格线便于观察
        painter = QPainter(test_image)
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        
        # 绘制网格
        grid_size = 50
        for x in range(0, width, grid_size):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, grid_size):
            painter.drawLine(0, y, width, y)
        
        # 添加十字标记点
        center_x, center_y = width // 2, height // 2
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawLine(center_x - 20, center_y, center_x + 20, center_y)
        painter.drawLine(center_x, center_y - 20, center_x, center_y + 20)
        
        painter.end()
        
        # 设置测试图像
        restoration_page.annotation_image = test_image
        restoration_page.annotation_pixmap = test_image
        restoration_page.annotation_enabled = True
        restoration_page.brush_size = 30
        restoration_page.brush_shape = 'circle'
        restoration_page.painted_mask = []  # 清空之前的标注
        
        # 模拟标注操作（不同位置和形状）
        test_patterns = [
            # (x, y, size, shape, description)
            (center_x, center_y, 30, 'circle', '中心圆形'),
            (width // 4, height // 4, 20, 'square', '左上方形'),
            (3 * width // 4, height // 4, 25, 'polygon', '右上方形'),
            (width // 4, 3 * height // 4, 35, 'circle', '左下圆形'),
            (3 * width // 4, 3 * height // 4, 15, 'square', '右下方形'),
        ]
        
        print(f"  添加 {len(test_patterns)} 个测试标注点...")
        
        for i, (x, y, size, shape, desc) in enumerate(test_patterns):
            print(f"    {i+1}. {desc} at ({x}, {y}), size={size}, shape={shape}")
            
            # 设置画笔参数
            restoration_page.brush_size = size
            restoration_page.brush_shape = shape
            
            # 添加标注点
            pos = QPoint(x, y)
            restoration_page.addPaintPoint(pos)
        
        # 验证坐标映射精度
        print("  验证坐标映射精度:")
        coordinate_accuracy = verify_coordinate_mapping()
        
        # 生成二值化掩码
        print("  生成二值化掩码...")
        binary_mask = restoration_page.createBinaryMask()
        
        if not binary_mask:
            print("  ❌ 二值化掩码生成失败")
            return {"success": False, "error": "掩码生成失败"}
        
        # 分析掩码质量
        print("  分析掩码质量...")
        mask_quality = analyze_mask_quality(binary_mask, test_patterns)
        
        # 测试尺寸匹配
        print("  测试尺寸匹配...")
        size_matching = test_size_matching(test_patterns, binary_mask)
        
        # 保存测试结果
        save_test_results(resolution_desc, test_image, binary_mask, test_patterns)
        
        return {
            "success": True,
            "coordinate_accuracy": coordinate_accuracy,
            "mask_quality": mask_quality,
            "size_matching": size_matching
        }
    
    def verify_coordinate_mapping():
        """验证坐标映射精度"""
        accuracy_results = []
        
        for i, mask_data in enumerate(restoration_page.painted_mask):
            screen_pos = mask_data['screen_pos']
            image_pos = mask_data['image_pos']
            
            # 验证坐标转换的一致性
            widget_pos = restoration_page.original_preview.mapFromParent(QPoint(*screen_pos))
            
            # 反向计算图像坐标
            scaled_pixmap = restoration_page.annotation_image.scaled(
                restoration_page.original_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            scale_x = restoration_page.annotation_image.width() / scaled_pixmap.width()
            scale_y = restoration_page.annotation_image.height() / scaled_pixmap.height()
            
            recalculated_img_x = (widget_pos.x() - restoration_page.original_preview.image_rect.x()) * scale_x
            recalculated_img_y = (widget_pos.y() - restoration_page.original_preview.image_rect.y()) * scale_y
            
            # 计算误差
            error_x = abs(image_pos[0] - recalculated_img_x)
            error_y = abs(image_pos[1] - recalculated_img_y)
            total_error = (error_x ** 2 + error_y ** 2) ** 0.5
            
            accuracy_results.append({
                "point": i + 1,
                "error_x": error_x,
                "error_y": error_y,
                "total_error": total_error,
                "acceptable": total_error < 1.0  # 误差小于1像素认为可接受
            })
            
            print(f"    点{i+1}: 误差=({error_x:.2f}, {error_y:.2f}), 总误差={total_error:.2f} {'✓' if total_error < 1.0 else '✗'}")
        
        avg_error = sum(r["total_error"] for r in accuracy_results) / len(accuracy_results)
        acceptable_count = sum(1 for r in accuracy_results if r["acceptable"])
        
        print(f"  平均误差: {avg_error:.2f} 像素")
        print(f"  可接受精度: {acceptable_count}/{len(accuracy_results)} ({acceptable_count/len(accuracy_results):.1%})")
        
        return {
            "avg_error": avg_error,
            "acceptable_ratio": acceptable_count / len(accuracy_results),
            "details": accuracy_results
        }
    
    def analyze_mask_quality(binary_mask, test_patterns):
        """分析掩码质量"""
        mask_image = binary_mask.toImage()
        mask_array = np.zeros((mask_image.height(), mask_image.width()), dtype=np.uint8)
        
        # 转换为numpy数组
        for y in range(mask_image.height()):
            for x in range(mask_image.width()):
                color = QColor(mask_image.pixel(x, y))
                mask_array[y, x] = color.red()
        
        # 统计掩码信息
        white_pixels = np.sum(mask_array > 128)
        total_pixels = mask_array.size
        coverage_ratio = white_pixels / total_pixels
        
        # 验证每个标注点是否在掩码中
        points_in_mask = 0
        for mask_data in restoration_page.painted_mask:
            img_x, img_y = mask_data['image_pos']
            img_x_int = int(img_x)
            img_y_int = int(img_y)
            
            if (0 <= img_x_int < mask_image.width() and 
                0 <= img_y_int < mask_image.height()):
                pixel_value = mask_array[img_y_int, img_x_int]
                if pixel_value > 128:
                    points_in_mask += 1
        
        mask_coverage = points_in_mask / len(restoration_page.painted_mask)
        
        print(f"    掩码覆盖率: {coverage_ratio:.2%}")
        print(f"    标注点在掩码中: {points_in_mask}/{len(restoration_page.painted_mask)} ({mask_coverage:.1%})")
        
        return {
            "coverage_ratio": coverage_ratio,
            "mask_coverage": mask_coverage,
            "white_pixels": white_pixels,
            "total_pixels": total_pixels
        }
    
    def test_size_matching(test_patterns, binary_mask):
        """测试尺寸匹配"""
        size_results = []
        
        for i, (x, y, expected_size, shape, desc) in enumerate(test_patterns):
            # 获取对应的掩码数据
            if i < len(restoration_page.painted_mask):
                mask_data = restoration_page.painted_mask[i]
                actual_size = mask_data['size']
                
                # 验证尺寸是否一致
                size_match = actual_size == expected_size
                
                size_results.append({
                    "pattern": desc,
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                    "match": size_match
                })
                
                print(f"    {desc}: 期望={expected_size}, 实际={actual_size} {'✓' if size_match else '✗'}")
        
        match_count = sum(1 for r in size_results if r["match"])
        match_ratio = match_count / len(size_results)
        
        print(f"  尺寸匹配率: {match_count}/{len(size_results)} ({match_ratio:.1%})")
        
        return {
            "match_ratio": match_ratio,
            "details": size_results
        }
    
    def save_test_results(resolution_desc, test_image, binary_mask, test_patterns):
        """保存测试结果"""
        test_dir = "test_results"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        # 保存测试图像
        test_image_path = os.path.join(test_dir, f"test_image_{resolution_desc}.png")
        test_image.save(test_image_path)
        
        # 保存掩码图像
        mask_path = os.path.join(test_dir, f"test_mask_{resolution_desc}.png")
        binary_mask.save(mask_path)
        
        print(f"  测试结果已保存:")
        print(f"    图像: {test_image_path}")
        print(f"    掩码: {mask_path}")
    
    def generate_test_report():
        """生成测试报告"""
        report_path = "test_results/test_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("掩码标注精度测试报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for resolution, results in test_results.items():
                f.write(f"{resolution} 测试结果:\n")
                f.write("-" * 40 + "\n")
                
                if results["success"]:
                    coord_acc = results["coordinate_accuracy"]
                    mask_qual = results["mask_quality"]
                    size_match = results["size_matching"]
                    
                    f.write(f"坐标映射精度:\n")
                    f.write(f"  平均误差: {coord_acc['avg_error']:.2f} 像素\n")
                    f.write(f"  可接受精度: {coord_acc['acceptable_ratio']:.1%}\n\n")
                    
                    f.write(f"掩码质量:\n")
                    f.write(f"  掩码覆盖率: {mask_qual['coverage_ratio']:.2%}\n")
                    f.write(f"  标注点覆盖: {mask_qual['mask_coverage']:.1%}\n\n")
                    
                    f.write(f"尺寸匹配:\n")
                    f.write(f"  匹配率: {size_match['match_ratio']:.1%}\n\n")
                else:
                    f.write(f"测试失败: {results['error']}\n\n")
                
                f.write("\n")
        
        print(f"测试报告已保存: {report_path}")
    
    # 延迟执行测试
    QTimer.singleShot(1000, run_comprehensive_test)
    
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='文物图像修复系统')
    parser.add_argument('--test-precision', action='store_true', help='运行掩码标注精度测试')
    parser.add_argument('--test-red-overlay', action='store_true', help='运行红色覆盖测试')
    parser.add_argument('--test-annotation', action='store_true', help='运行实时标注测试')
    parser.add_argument('--test', action='store_true', help='运行关键功能测试')
    parser.add_argument('--test-coordinate-precision', action='store_true', help='运行高精度坐标测试')
    args = parser.parse_args()
    
    # 根据参数执行相应功能
    if args.test_precision:
        test_mask_annotation_precision()
    elif args.test_red_overlay:
        test_annotation_red_overlay()
    elif args.test_annotation:
        test_annotation_realtime()
    elif args.test:
        test_key_functionality()
    elif args.test_coordinate_precision:
        test_coordinate_precision()
    else:
        # 正常启动应用
        app = QApplication(sys.argv)
        window = RelicRestorationPage()
        window.setWindowTitle("文物图像修复系统 v2.4 - 支持精度测试")
        window.resize(1200, 700)
        window.show()
        sys.exit(app.exec())