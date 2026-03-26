import sys
import os
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSlider, QColorDialog, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QMouseEvent


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initVariables()

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('图像处理应用程序')
        self.setGeometry(100, 100, 1200, 800)

        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        control_panel = self.createControlPanel()
        main_layout.addWidget(control_panel, 1)

        # 右侧图片展示区域
        self.image_display = ImageDisplayWidget()
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.image_display, 3)

        # 连接信号槽
        self.connectSignals()

    def createControlPanel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 文件操作区域
        file_group = QWidget()
        file_layout = QVBoxLayout(file_group)

        self.upload_btn = QPushButton('上传图片')
        self.upload_btn.setMinimumHeight(40)
        file_layout.addWidget(self.upload_btn)

        self.confirm_btn = QPushButton('确认生成二值图')
        self.confirm_btn.setMinimumHeight(40)
        self.confirm_btn.setEnabled(False)
        file_layout.addWidget(self.confirm_btn)

        self.save_btn = QPushButton('保存二值图')
        self.save_btn.setMinimumHeight(40)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton('重置')
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.setEnabled(False)
        file_layout.addWidget(self.reset_btn)

        layout.addWidget(file_group)

        # 涂抹工具控制区域
        tools_group = QWidget()
        tools_layout = QVBoxLayout(tools_group)

        # 画笔大小控制
        brush_size_label = QLabel('画笔大小:')
        tools_layout.addWidget(brush_size_label)

        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setRange(1, 70)
        self.brush_size_slider.setValue(5)
        self.brush_size_slider.valueChanged.connect(self.updateBrushSize)
        tools_layout.addWidget(self.brush_size_slider)

        self.brush_size_value = QLabel('5')
        tools_layout.addWidget(self.brush_size_value)

        # 颜色选择
        self.color_btn = QPushButton('选择画笔颜色')
        self.color_btn.setMinimumHeight(40)
        tools_layout.addWidget(self.color_btn)

        self.color_preview = QLabel()
        self.color_preview.setFixedSize(50, 50)
        self.color_preview.setStyleSheet('background-color: red; border: 1px solid black;')
        tools_layout.addWidget(self.color_preview)

        # 画笔形状选择
        brush_shape_label = QLabel('画笔形状:')
        tools_layout.addWidget(brush_shape_label)

        from PyQt5.QtWidgets import QComboBox
        self.brush_shape_combo = QComboBox()
        self.brush_shape_combo.addItems(['圆形', '方形', '多边形'])
        self.brush_shape_combo.setCurrentIndex(0)  # 默认圆形
        self.brush_shape_combo.currentIndexChanged.connect(self.updateBrushShape)
        tools_layout.addWidget(self.brush_shape_combo)

        layout.addWidget(tools_group)

        # 状态信息
        self.status_label = QLabel('状态: 等待上传图片')
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        return panel

    def initVariables(self):
        """初始化变量"""
        self.current_image = None
        self.original_pixmap = None
        self.current_binary_image = None
        self.brush_color = QColor(255, 0, 0)  # 默认红色
        self.brush_size = 5
        self.brush_shape = 'circle'  # 默认圆形
        self.binary_image_generated = False

    def connectSignals(self):
        """连接信号槽"""
        self.upload_btn.clicked.connect(self.uploadImage)
        self.confirm_btn.clicked.connect(self.generateBinaryImage)
        self.reset_btn.clicked.connect(self.resetImage)
        self.color_btn.clicked.connect(self.selectBrushColor)
        self.save_btn.clicked.connect(self.saveBinaryImage)

    def uploadImage(self):
        """上传图片功能"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择图片文件',
            '',
            '图片文件 (*.jpg *.jpeg *.png *.bmp *.gif);;所有文件 (*)'
        )

        if file_path:
            try:
                # 加载图片
                self.original_pixmap = QPixmap(file_path)
                if not self.original_pixmap.isNull():
                    self.current_image = self.original_pixmap.copy()
                    self.image_display.setImage(self.current_image)
                    self.image_display.enableDrawing(True)
                    self.image_display.setBrushShape(self.brush_shape)

                    # 启用相关按钮
                    self.confirm_btn.setEnabled(True)
                    self.reset_btn.setEnabled(True)

                    # 重置保存按钮状态
                    self.save_btn.setEnabled(False)
                    self.binary_image_generated = False
                    self.current_binary_image = None

                    self.status_label.setText(f'状态: 已加载图片 - {os.path.basename(file_path)}')
                else:
                    QMessageBox.warning(self, '错误', '无法加载图片文件')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'加载图片时发生错误: {str(e)}')

    def selectBrushColor(self):
        """选择画笔颜色"""
        color = QColorDialog.getColor(self.brush_color, self, '选择画笔颜色')
        if color.isValid():
            self.brush_color = color
            self.color_preview.setStyleSheet(f'background-color: {color.name()}; border: 1px solid black;')
            self.image_display.setBrushColor(color)
            self.image_display.setBrushShape(self.brush_shape)

    def updateBrushSize(self, value):
        """更新画笔大小"""
        self.brush_size = value
        self.brush_size_value.setText(str(value))
        self.image_display.setBrushSize(value)

    def updateBrushShape(self, index):
        """更新画笔形状"""
        shapes = ['circle', 'square', 'polygon']
        self.brush_shape = shapes[index]
        self.image_display.setBrushShape(self.brush_shape)

    def generateBinaryImage(self):
        """生成二值化图像"""
        if self.current_image is None:
            return

        try:
            # 获取涂抹区域
            painted_mask = self.image_display.getPaintedMask()

            if painted_mask is None:
                QMessageBox.information(self, '提示', '请先在图片上进行涂抹操作')
                return

            # 创建二值化图像
            binary_image = self.createBinaryImage(painted_mask)

            # 显示二值化图像
            self.image_display.setImage(binary_image)
            self.image_display.enableDrawing(False)

            # 保存二值化图像引用
            self.current_binary_image = binary_image
            self.binary_image_generated = True

            # 启用保存按钮
            self.save_btn.setEnabled(True)

            self.status_label.setText('状态: 二值化图像生成完成，可以保存')
            QMessageBox.information(self, '成功', '二值化图像已生成，现在可以保存到本地')

        except Exception as e:
            QMessageBox.critical(self, '错误', f'生成二值化图像时发生错误: {str(e)}')

    def createBinaryImage(self, painted_mask):
        """创建二值化图像 - 支持不同形状"""
        # 创建黑色背景
        binary_pixmap = QPixmap(self.original_pixmap.size())
        binary_pixmap.fill(Qt.black)

        # 将涂抹区域设置为白色
        painter = QPainter(binary_pixmap)
        painter.setPen(QPen(Qt.white, 1))
        painter.setBrush(QBrush(Qt.white))
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制涂抹区域
        for mask_data in painted_mask:
            if isinstance(mask_data, dict):
                # 新格式：包含位置和形状信息
                img_x, img_y = mask_data['image_pos']
                brush_size = mask_data['size']
                shape = mask_data['shape']

                if shape == 'circle':
                    painter.drawEllipse(int(img_x - brush_size // 2), int(img_y - brush_size // 2),
                                        brush_size, brush_size)
                elif shape == 'square':
                    painter.drawRect(int(img_x - brush_size // 2), int(img_y - brush_size // 2),
                                     brush_size, brush_size)
                elif shape == 'polygon':
                    # 绘制六边形
                    points = []
                    for i in range(6):
                        angle = i * 60 * math.pi / 180
                        px = img_x + brush_size // 2 * math.cos(angle)
                        py = img_y + brush_size // 2 * math.sin(angle)
                        points.append(QPoint(int(px), int(py)))
                    painter.drawPolygon(points)
            else:
                # 兼容旧格式（矩形）
                painter.drawRect(mask_data)

        painter.end()

        return binary_pixmap

    def resetImage(self):
        """重置图像"""
        if self.original_pixmap is not None:
            self.current_image = self.original_pixmap.copy()
            self.image_display.setImage(self.current_image)
            self.image_display.enableDrawing(True)
            self.image_display.clearPaintedMask()

            # 重置二值图相关状态
            self.current_binary_image = None
            self.binary_image_generated = False
            self.save_btn.setEnabled(False)

            # 重新设置画笔形状
            self.image_display.setBrushShape(self.brush_shape)

            self.status_label.setText('状态: 图像已重置')

    def saveBinaryImage(self):
        """保存二值化图像到本地"""
        if not self.binary_image_generated or self.current_binary_image is None:
            QMessageBox.warning(self, '警告', '请先生成二值化图像')
            return

        try:
            # 生成默认文件名
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"binary_image_{timestamp}.png"

            # 打开文件保存对话框
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                '保存二值化图像',
                default_filename,
                'PNG文件 (*.png);;JPEG文件 (*.jpg *.jpeg);;BMP文件 (*.bmp);;所有文件 (*)',
                'PNG文件 (*.png)'
            )

            if file_path:
                # 确保文件扩展名正确
                if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    if 'PNG' in selected_filter:
                        file_path += '.png'
                    elif 'JPEG' in selected_filter:
                        file_path += '.jpg'
                    elif 'BMP' in selected_filter:
                        file_path += '.bmp'

                # 检查文件权限和目录存在性
                directory = os.path.dirname(file_path)
                if directory and not os.path.exists(directory):
                    try:
                        os.makedirs(directory, exist_ok=True)
                    except OSError as e:
                        QMessageBox.critical(self, '错误', f'无法创建目录: {str(e)}')
                        return

                # 保存图像，保持高质量
                if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                    # JPEG格式，设置高质量
                    quality = 95  # 0-100，95为高质量
                    success = self.current_binary_image.save(file_path, 'JPEG', quality)
                elif file_path.lower().endswith('.png'):
                    # PNG格式，无损压缩
                    success = self.current_binary_image.save(file_path, 'PNG')
                elif file_path.lower().endswith('.bmp'):
                    # BMP格式
                    success = self.current_binary_image.save(file_path, 'BMP')
                else:
                    # 默认PNG格式
                    success = self.current_binary_image.save(file_path, 'PNG')

                if success:
                    self.status_label.setText(f'状态: 二值化图像已保存 - {os.path.basename(file_path)}')
                    QMessageBox.information(
                        self,
                        '保存成功',
                        f'二值化图像已成功保存到:\n{file_path}\n\n文件大小: {self.getFileSize(file_path)}'
                    )
                else:
                    QMessageBox.critical(self, '保存失败', '图像保存过程中发生未知错误')

        except PermissionError:
            QMessageBox.critical(
                self,
                '权限错误',
                '没有足够的权限写入指定位置。\n请选择其他保存位置或检查文件权限。'
            )
        except Exception as e:
            QMessageBox.critical(self, '保存错误', f'保存图像时发生错误:\n{str(e)}')

    def getFileSize(self, file_path):
        """获取文件大小的友好显示"""
        try:
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024:
                return f"{size_bytes} 字节"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        except:
            return "未知大小"


class ImageDisplayWidget(QWidget):
    """图片显示和绘制组件"""

    def __init__(self):
        super().__init__()
        self.image_pixmap = None
        self.drawing_enabled = False
        self.brush_color = QColor(255, 0, 0)
        self.brush_size = 5
        self.brush_shape = 'circle'  # 默认圆形
        self.painting = False
        self.painted_mask = []
        self.scale_factor = 1.0
        self.image_rect = QRect()  # 存储图片在窗口中的实际显示区域

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
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), Qt.lightGray)

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

            # 绘制涂抹区域
            if self.drawing_enabled and self.painted_mask:
                painter.setPen(QPen(self.brush_color, 2))
                painter.setBrush(QBrush(self.brush_color))

                for mask_data in self.painted_mask:
                    if isinstance(mask_data, dict):
                        # 新的格式：包含位置和形状信息
                        screen_x, screen_y = mask_data['screen_pos']
                        brush_size = mask_data['size']
                        shape = mask_data['shape']

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
                                px = screen_x + brush_size // 2 * cos(angle)
                                py = screen_y + brush_size // 2 * sin(angle)
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
        """添加涂抹点 - 修复坐标映射问题"""
        if self.image_pixmap is None:
            return

        # 检查点击位置是否在图片显示范围内
        if not self.image_rect.contains(pos):
            return

        # 计算缩放比例
        scaled_pixmap = self.image_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        scale_x = self.image_pixmap.width() / scaled_pixmap.width()
        scale_y = self.image_pixmap.height() / scaled_pixmap.height()

        # 精确映射到图片坐标
        img_x = (pos.x() - self.image_rect.x()) * scale_x
        img_y = (pos.y() - self.image_rect.y()) * scale_y

        # 确保坐标在图片范围内
        img_x = max(0, min(img_x, self.image_pixmap.width() - 1))
        img_y = max(0, min(img_y, self.image_pixmap.height() - 1))

        # 存储涂抹数据（包含屏幕坐标和图片坐标）
        mask_data = {
            'screen_pos': (pos.x(), pos.y()),  # 屏幕坐标用于绘制
            'image_pos': (img_x, img_y),  # 图片坐标用于二值化
            'size': self.brush_size,
            'shape': self.brush_shape
        }

        self.painted_mask.append(mask_data)
        self.update()


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 创建主窗口
    window = ImageProcessingApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()  # trtea