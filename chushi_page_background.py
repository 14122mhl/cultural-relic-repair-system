import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QFont, QFontDatabase


class FunctionButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self, title, description, parent=None):
        super().__init__(parent)
        self.initUI(title, description)

    def initUI(self, title, description):
        self.setMinimumSize(350, 300)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title_label = QLabel(title)
        title_font = QFont('SimHei', 22, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #ffffff; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);")
        layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setFont(QFont('SimHei', 12))
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #f0f0f0; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);")
        layout.addWidget(desc_label)

        enter_btn = QPushButton("进入功能")
        enter_btn.setFont(QFont('SimHei', 14, QFont.Weight.Medium))
        enter_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(59, 130, 246, 180);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: rgba(37, 99, 235, 200);
            }
            QPushButton:pressed {
                background-color: rgba(29, 78, 216, 200);
            }
        """)
        layout.addWidget(enter_btn)

        self.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 80);
                border-radius: 12px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
            }
            QWidget:hover {
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
                transform: translateY(-5px);
                background-color: rgba(255, 255, 255, 100);
            }
        """)

        enter_btn.clicked.connect(self.on_clicked)
        self.clicked.connect(self.button_animation)

    def on_clicked(self):
        self.clicked.emit()

    def button_animation(self):
        animation = QPropertyAnimation(self, b"geometry")
        animation.setDuration(200)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        original_geometry = self.geometry()
        animation.setStartValue(original_geometry)
        animation.setKeyValueAt(0.5, original_geometry.adjusted(8, 8, -8, -8))
        animation.setEndValue(original_geometry)

        animation.start()

    def mousePressEvent(self, event):
        self.on_clicked()
        super().mousePressEvent(event)


# 保持类名与原模块一致，确保替换时无需修改其他代码
class InitialInterface(QMainWindow):
    switch_to_repair = pyqtSignal()
    switch_to_dataset = pyqtSignal()

    def __init__(self):
        super().__init__()
        # 字体设置保持不变
        font_id = QFontDatabase.addApplicationFont("SimHei.ttf")
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            app_font = QFont(font_family)
            QApplication.setFont(app_font)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('文物图像修复系统')
        self.setMinimumSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 背景图片设置
        central_widget.setStyleSheet(f"""
            background-image: url("D:/background/feitian.jpg");
            background-repeat: no-repeat;
            background-size: cover;
            background-position: center;
        """)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(60, 60, 60, 60)
        main_layout.setSpacing(50)

        # 标题样式优化，适应背景
        title_label = QLabel('文物图像修复系统')
        title_font = QFont('SimHei', 32, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            color: #ffffff;
            text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.8);
            padding: 20px;
            margin-bottom: 20px;
        """)
        main_layout.addWidget(title_label)

        # 副标题样式优化
        subtitle_label = QLabel('请选择所需功能')
        subtitle_font = QFont('SimHei', 16)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet(
            "color: #f0f0f0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8); margin-bottom: 20px;")
        main_layout.addWidget(subtitle_label)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(60)
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addStretch()
        main_layout.addLayout(buttons_layout)
        main_layout.addStretch()

        self.repair_btn = FunctionButton(
            "图像修复",
            "对受损文物图像进行修复和复原处理"
        )
        self.repair_btn.clicked.connect(self.on_repair_clicked)
        buttons_layout.addWidget(self.repair_btn)

        self.dataset_btn = FunctionButton(
            "数据集生成",
            "创建和扩充文物图像数据集用于模型训练"
        )
        self.dataset_btn.clicked.connect(self.on_dataset_clicked)
        buttons_layout.addWidget(self.dataset_btn)

        self.center_window()

    def on_repair_clicked(self):
        self.switch_to_repair.emit()

    def on_dataset_clicked(self):
        self.switch_to_dataset.emit()

    def center_window(self):
        frame_geometry = self.frameGeometry()
        center_point = QApplication.primaryScreen().availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InitialInterface()
    window.show()
    sys.exit(app.exec())
