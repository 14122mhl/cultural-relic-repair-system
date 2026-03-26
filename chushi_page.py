
from PyQt5 import QtCore, QtGui, QtWidgets
import os

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1055,100)
        Form.setAutoFillBackground(True)

        self.set_background_image(Form, "D:/python_chuangxin/jiemianpythonProject/UI/fengmian.jpg")

        # 数据集生成按钮（位置：480,492，大小161x35）
        self.pushButton = QtWidgets.QPushButton(parent=Form)
        self.pushButton.setGeometry(QtCore.QRect(480, 492, 161, 35))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: #FFFFCC;")
        self.pushButton.setObjectName("pushButton")

        # 图像修复按钮（位置：480,542，大小161x35）
        self.pushButton_2 = QtWidgets.QPushButton(parent=Form)
        self.pushButton_2.setGeometry(QtCore.QRect(480, 542, 161, 35))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color: #FFFFCC;")
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def set_background_image(self, widget, image_path):
        if not os.path.exists(image_path):
            print(f"警告: 图片文件不存在 - {image_path}")
            return

        pixmap = QtGui.QPixmap(image_path)
        if pixmap.isNull():
            print(f"警告: 无法加载图片 - {image_path}")
            return

        scaled_pixmap = pixmap.scaled(
            widget.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )

        palette = QtGui.QPalette()
        palette.setBrush(
            QtGui.QPalette.ColorGroup.All,
            QtGui.QPalette.ColorRole.Window,
            QtGui.QBrush(scaled_pixmap)
        )
        widget.setPalette(palette)
        widget.setAutoFillBackground(True)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "文物图像修复系统"))  # 统一窗口标题
        self.pushButton.setText(_translate("Form", "数据集生成"))
        self.pushButton_2.setText(_translate("Form", "图像修复"))


# 关键：定义可导入的 InitialInterface 类（与 main.py 导入的名称匹配）
class InitialInterface(QtWidgets.QWidget):
    # 保留原接口信号，确保与修复界面兼容
    switch_to_repair = QtCore.pyqtSignal()    # 图像修复切换信号
    switch_to_dataset = QtCore.pyqtSignal()  # 数据集生成切换信号

    def __init__(self):
        super().__init__()
        # 初始化 UI 布局
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # 绑定按钮点击事件到信号
        self._bind_buttons()
        # 窗口居中显示
        self._center_window()

    def _bind_buttons(self):
        # 数据集生成按钮 → 触发 switch_to_dataset 信号
        self.ui.pushButton.clicked.connect(self.switch_to_dataset.emit)
        # 图像修复按钮 → 触发 switch_to_repair 信号
        self.ui.pushButton_2.clicked.connect(self.switch_to_repair.emit)

    def _center_window(self):
        # 获取屏幕中心，让窗口居中
        screen_geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        window_geo = self.frameGeometry()
        window_geo.moveCenter(screen_geo.center())
        self.move(window_geo.topLeft())