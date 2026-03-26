import cv2 as cv
import numpy as np
import logging
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap  # 新增导入QPixmap

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageParameterAdjuster:
    """图像参数调整器，实现对比度增强、亮度调整和色彩平衡功能"""

    def __init__(self):
        # 默认参数值
        self.contrast_factor = 1.0  # 对比度因子，1.0为原始
        self.brightness_value = 0  # 亮度值，0为原始
        self.color_balance_weights = [1.0, 1.0, 1.0]  # 色彩平衡权重（B, G, R）

        # 效果强度范围
        self.min_contrast = 0.1
        self.max_contrast = 3.0
        self.min_brightness = -100
        self.max_brightness = 100
        self.min_color_weight = 0.1
        self.max_color_weight = 2.0

        logger.info("图像参数调整器初始化完成")

    def set_contrast(self, factor):
        """设置对比度因子，确保在有效范围内"""
        if self.min_contrast <= factor <= self.max_contrast:
            self.contrast_factor = factor
            logger.info(f"设置对比度: {factor:.2f}")
            return True
        logger.warning(f"对比度值 {factor} 超出范围 [{self.min_contrast}, {self.max_contrast}]")
        return False

    def set_brightness(self, value):
        """设置亮度值，确保在有效范围内"""
        if self.min_brightness <= value <= self.max_brightness:
            self.brightness_value = value
            logger.info(f"设置亮度: {value}")
            return True
        logger.warning(f"亮度值 {value} 超出范围 [{self.min_brightness}, {self.max_brightness}]")
        return False

    def set_color_balance(self, b_weight, g_weight, r_weight):
        """设置色彩平衡权重，确保在有效范围内"""
        weights = [b_weight, g_weight, r_weight]
        valid = all(self.min_color_weight <= w <= self.max_color_weight for w in weights)

        if valid:
            self.color_balance_weights = weights
            logger.info(f"设置色彩平衡: B={b_weight:.2f}, G={g_weight:.2f}, R={r_weight:.2f}")
            return True

        logger.warning(
            f"色彩平衡值超出范围 [{self.min_color_weight}, {self.max_color_weight}]: "
            f"B={b_weight}, G={g_weight}, R={r_weight}"
        )
        return False

    def adjust_contrast(self, image):
        """
        增强图像对比度
        参数: image - BGR格式的numpy数组
        返回: 调整后的BGR图像
        """
        if self.contrast_factor == 1.0:
            return image.copy()

        try:
            # 转换为浮点数处理以避免溢出
            img_float = image.astype(np.float32)
            # 对比度调整公式: out = (in - 127.5) * contrast + 127.5
            adjusted = (img_float - 127.5) * self.contrast_factor + 127.5
            # 裁剪到0-255范围并转换回uint8
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            logger.debug("对比度调整完成")
            return adjusted
        except Exception as e:
            logger.error(f"对比度调整失败: {str(e)}")
            return image.copy()

    def adjust_brightness(self, image):
        """
        调整图像亮度
        参数: image - BGR格式的numpy数组
        返回: 调整后的BGR图像
        """
        if self.brightness_value == 0:
            return image.copy()

        try:
            # 转换为浮点数处理以避免溢出
            img_float = image.astype(np.float32)
            # 亮度调整公式: out = in + brightness
            adjusted = img_float + self.brightness_value
            # 裁剪到0-255范围并转换回uint8
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            logger.debug("亮度调整完成")
            return adjusted
        except Exception as e:
            logger.error(f"亮度调整失败: {str(e)}")
            return image.copy()

    def adjust_color_balance(self, image):
        """
        调整图像色彩平衡
        参数: image - BGR格式的numpy数组
        返回: 调整后的BGR图像
        """
        if all(w == 1.0 for w in self.color_balance_weights):
            return image.copy()

        try:
            # 分离通道
            b, g, r = cv.split(image.astype(np.float32))

            # 应用色彩平衡权重
            b = b * self.color_balance_weights[0]
            g = g * self.color_balance_weights[1]
            r = r * self.color_balance_weights[2]

            # 合并通道并裁剪到0-255范围
            adjusted = cv.merge([b, g, r])
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            logger.debug("色彩平衡调整完成")
            return adjusted
        except Exception as e:
            logger.error(f"色彩平衡调整失败: {str(e)}")
            return image.copy()

    def apply_all_adjustments(self, image, apply_contrast=True, apply_brightness=True, apply_color=True):
        """
        应用所有选中的调整
        参数:
            image - BGR格式的numpy数组
            apply_contrast - 是否应用对比度调整
            apply_brightness - 是否应用亮度调整
            apply_color - 是否应用色彩平衡
        返回: 调整后的BGR图像
        """
        if image is None:
            logger.warning("无法处理空图像")
            return None

        try:
            adjusted = image.copy()

            # 按顺序应用调整
            if apply_brightness:
                adjusted = self.adjust_brightness(adjusted)
            if apply_contrast:
                adjusted = self.adjust_contrast(adjusted)
            if apply_color:
                adjusted = self.adjust_color_balance(adjusted)

            logger.info("所有选中的图像调整已应用")
            return adjusted
        except Exception as e:
            logger.error(f"应用图像调整失败: {str(e)}")
            return image.copy()

    def bgr_to_qimage(self, bgr_image):
        """
        将BGR格式的numpy数组转换为QImage
        参数: bgr_image - BGR格式的numpy数组
        返回: QImage对象
        """
        if bgr_image is None:
            return None

        try:
            # 转换为RGB格式
            rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            # 创建QImage
            q_image = QImage(
                rgb_image.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888
            )
            return q_image
        except Exception as e:
            logger.error(f"BGR转QImage失败: {str(e)}")
            return None

    def get_adjustment_params(self):
        """获取当前调整参数"""
        return {
            "contrast": self.contrast_factor,
            "brightness": self.brightness_value,
            "color_balance": {
                "blue": self.color_balance_weights[0],
                "green": self.color_balance_weights[1],
                "red": self.color_balance_weights[2]
            }
        }

    def reset_parameters(self):
        """重置所有参数为默认值"""
        self.contrast_factor = 1.0
        self.brightness_value = 0
        self.color_balance_weights = [1.0, 1.0, 1.0]
        logger.info("所有参数已重置为默认值")


# 以下是与主界面集成的辅助函数
def integrate_parameter_adjuster(main_window):
    """
    将参数调整器集成到主界面
    参数: main_window - 主界面窗口对象
    """
    try:
        # 创建参数调整器实例并附加到主窗口
        main_window.param_adjuster = ImageParameterAdjuster()

        # 修改主窗口的参数处理函数
        def on_contrast_changed(state):
            main_window.contrast_enhance = state == Qt.CheckState.Checked.value
            # 应用默认对比度增强（1.5倍）
            if main_window.contrast_enhance:
                main_window.param_adjuster.set_contrast(1.5)
            else:
                main_window.param_adjuster.set_contrast(1.0)
            main_window.log_message(f"对比度增强: {'开启' if main_window.contrast_enhance else '关闭'}")
            # 如果已有修复结果，实时更新显示
            update_preview_with_parameters(main_window)

        def on_brightness_changed(state):
            main_window.brightness_adjust = state == Qt.CheckState.Checked.value
            # 应用默认亮度调整（+30）
            if main_window.brightness_adjust:
                main_window.param_adjuster.set_brightness(30)
            else:
                main_window.param_adjuster.set_brightness(0)
            main_window.log_message(f"亮度调整: {'开启' if main_window.brightness_adjust else '关闭'}")
            # 如果已有修复结果，实时更新显示
            update_preview_with_parameters(main_window)

        def on_color_balance_changed(state):
            main_window.color_balance = state == Qt.CheckState.Checked.value
            # 应用默认色彩平衡（增强红色和绿色通道）
            if main_window.color_balance:
                main_window.param_adjuster.set_color_balance(0.9, 1.2, 1.2)
            else:
                main_window.param_adjuster.set_color_balance(1.0, 1.0, 1.0)
            main_window.log_message(f"色彩平衡: {'开启' if main_window.color_balance else '关闭'}")
            # 如果已有修复结果，实时更新显示
            update_preview_with_parameters(main_window)

        # 重新连接信号槽
        main_window.contrast_checkbox.stateChanged.disconnect()
        main_window.brightness_checkbox.stateChanged.disconnect()
        main_window.color_balance_checkbox.stateChanged.disconnect()

        main_window.contrast_checkbox.stateChanged.connect(on_contrast_changed)
        main_window.brightness_checkbox.stateChanged.connect(on_brightness_changed)
        main_window.color_balance_checkbox.stateChanged.connect(on_color_balance_changed)

        logger.info("参数调整器已成功集成到主界面")
        return True

    except Exception as e:
        logger.error(f"参数调整器集成失败: {str(e)}")
        return False


def update_preview_with_parameters(main_window):
    """根据当前参数更新预览图像（修复QPixmap调用错误）"""
    try:
        if not main_window.restored_images or main_window.current_index >= len(main_window.restored_images):
            return

        # 获取当前修复图像（RGB格式）
        current_rgb = main_window.restored_images[main_window.current_index]
        # 转换为BGR格式进行处理
        current_bgr = cv.cvtColor(current_rgb, cv.COLOR_RGB2BGR)

        # 应用参数调整
        adjusted_bgr = main_window.param_adjuster.apply_all_adjustments(
            current_bgr,
            apply_contrast=main_window.contrast_enhance,
            apply_brightness=main_window.brightness_adjust,
            apply_color=main_window.color_balance
        )

        # 转换回QImage并显示（修复处：直接使用QPixmap，无需通过main_window.QtGui）
        q_image = main_window.param_adjuster.bgr_to_qimage(adjusted_bgr)
        if q_image:
            preview_size = main_window.restored_preview.size()
            scaled_image = q_image.scaled(
                preview_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            # 修复：直接使用QPixmap.fromImage，而非main_window.QtGui.QPixmap
            main_window.restored_preview.setPixmap(QPixmap.fromImage(scaled_image))

    except Exception as e:
        logger.error(f"更新预览图像失败: {str(e)}")