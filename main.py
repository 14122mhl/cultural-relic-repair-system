import sys
import os

# 1. 修正LAMA路径（指向实际模型所在的D盘路径）
lama_folder = r"D:\python_chuangxin\jiemianpythonProject\lama"  # 已修正为D盘实际路径
saicinpainting_folder = os.path.join(lama_folder, "saicinpainting")

# 2. 确保路径已添加到Python搜索路径
if lama_folder not in sys.path:
    sys.path.append(lama_folder)
if saicinpainting_folder not in sys.path:
    sys.path.append(saicinpainting_folder)

# 3. 验证saicinpainting导入
try:
    import saicinpainting
    print("✅ main.py中saicinpainting导入成功")
except ImportError as e:
    print(f"❌ saicinpainting导入失败: {e}")
    print(f"当前搜索路径: {sys.path}")
    sys.exit(1)


from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QProgressBar, QListWidget, QGroupBox,
                             QRadioButton, QComboBox, QMessageBox, QFrame, QCheckBox,
                             QListWidgetItem, QApplication, QToolTip, QLineEdit,
                             QMainWindow, QStackedWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPixmap, QImage
import cv2 as cv
import numpy as np
import time
import json

# 5. 导入predict.py（模型加载相关）
try:
    from predict import FastRestorationModel
    PREDICT_AVAILABLE = True
except ImportError as e:
    print(f"predict.py导入失败: {e}")
    print("快速修复将使用模拟模式")
    PREDICT_AVAILABLE = False

# 6. 页面相关导入（仅保留用到的页面，删除未使用的冗余导入）
from chushi_page import InitialInterface  # 适配已替换的初始界面（带两个按钮）
from xiufu import RelicRestorationPage  # 图像修复页面
from shujuji_mask import DatasetGeneratorPage  # 数据集生成页面


class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("文物图像修复系统")
        self.setMinimumSize(1200, 800)

        # 创建堆叠窗口用于页面切换（核心切换容器）
        self.stack_widget = QStackedWidget()
        self.setCentralWidget(self.stack_widget)

        # 初始化各个页面（与导入的页面类对应）
        self.initial_page = InitialInterface()  # 已替换的初始界面
        self.repair_page = RelicRestorationPage()  # 图像修复页面
        self.dataset_page = DatasetGeneratorPage()  # 数据集生成页面

        # 将页面添加到堆叠窗口（顺序对应切换索引，可按需调整）
        self.stack_widget.addWidget(self.initial_page)    # 索引0：初始页面
        self.stack_widget.addWidget(self.repair_page)     # 索引1：修复页面
        self.stack_widget.addWidget(self.dataset_page)    # 索引2：数据集页面

        # 连接页面切换信号（确保与各页面信号名匹配）
        # 初始页面 → 修复/数据集页面
        self.initial_page.switch_to_repair.connect(self.show_repair_page)
        self.initial_page.switch_to_dataset.connect(self.show_dataset_page)
        # 修复/数据集页面 → 初始页面
        self.repair_page.back_to_initial.connect(self.show_initial_page)
        self.dataset_page.back_to_initial.connect(self.show_initial_page)

    def show_initial_page(self):
        """切换显示初始页面（已替换的带两个按钮的界面）"""
        self.stack_widget.setCurrentWidget(self.initial_page)
        # 可选：切换时同步窗口大小（若初始页面有特定尺寸需求）
        self.setMinimumSize(1055, 700)  # 匹配chushi_page.py的初始尺寸

    def show_repair_page(self):
        """切换显示图像修复页面"""
        self.stack_widget.setCurrentWidget(self.repair_page)
        # 可选：切换时同步修复页面的推荐尺寸
        self.setMinimumSize(1200, 700)  # 适配修复页面的布局

    def show_dataset_page(self):
        """切换显示数据集生成页面"""
        self.stack_widget.setCurrentWidget(self.dataset_page)
        # 可选：根据数据集页面布局调整窗口最小尺寸
        self.setMinimumSize(1200, 700)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 确保中文正常显示（适配SimHei字体，避免乱码）
    font = QFont("SimHei")
    app.setFont(font)

    # 启动主控制器（所有页面的入口）
    window = MainController()
    window.show()
    sys.exit(app.exec())