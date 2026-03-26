from PyQt6 import QtCore, QtGui, QtWidgets
import os


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1055, 700)
        Form.setAutoFillBackground(True)

        self.set_background_image(Form, "D:/python_chuangxin/jiemianpythonProject/UI/fengmian.jpg")

        # 数据集生成按钮，向上微调8像素（从500调整为492）
        self.pushButton = QtWidgets.QPushButton(parent=Form)
        self.pushButton.setGeometry(QtCore.QRect(480, 492, 161, 35))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: #FFFFCC;")
        self.pushButton.setObjectName("pushButton")

        # 图像修复按钮，向上微调8像素（从550调整为542）
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
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "数据集生成"))
        self.pushButton_2.setText(_translate("Form", "图像修复"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec())
