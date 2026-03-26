import cv2
import numpy as np
import time
import os
from PyQt6.QtCore import QObject, pyqtSignal


class RestorationWorker(QObject):
    """修复功能工作类，与界面分离"""
    progress_updated = pyqtSignal(int, str)  # 进度值, 进度文本
    status_updated = pyqtSignal(str)  # 状态文本
    log_updated = pyqtSignal(str)  # 日志消息
    finished = pyqtSignal(object)  # 处理结果

    def __init__(self, images, strength, algorithm, denoise_level, detail_level, output_dir, format):
        super().__init__()
        self.images = [img.copy() for img in images]  # 原始图像列表
        self.strength = strength  # 修复强度
        self.algorithm = algorithm  # 算法类型：0-快速，1-精细，2-深度
        self.denoise_level = denoise_level  # 降噪程度
        self.detail_level = detail_level  # 细节保留程度
        self.output_dir = output_dir  # 输出目录
        self.format = format.lower()  # 保存格式
        self.canceled = False  # 是否取消处理

    def run(self):
        """执行修复处理"""
        start_time = time.time()
        stats = {
            'time': '00:00:00',
            'count': 0,
            'avg_area': 0
        }

        restored_images = []
        total_area = 0

        try:
            total = len(self.images)
            algo_name = ["快速修复", "精细修复", "深度修复"][self.algorithm]
            self.log_updated.emit(f"开始{algo_name}，共 {total} 张图像")

            for i, image in enumerate(self.images):
                if self.canceled:
                    self.finished.emit(None)
                    return

                # 更新进度和状态
                progress = int((i + 1) / total * 100)
                progress_text = f"正在处理 {i + 1}/{total} ({progress}%)"
                self.progress_updated.emit(progress, progress_text)
                self.status_updated.emit(f"正在修复 {i + 1}/{total}")
                self.log_updated.emit(f"开始修复第 {i + 1} 张图像")

                try:
                    # 检测图像损伤区域（模拟）
                    height, width = image.shape[:2]
                    damage_area = self.detect_damage_area(image)
                    total_area += damage_area

                    # 根据选择的算法进行修复
                    if self.algorithm == 0:  # 快速修复
                        restored_img = self.quick_restoration(image)
                    elif self.algorithm == 1:  # 精细修复
                        restored_img = self.fine_restoration(image)
                    else:  # 深度修复（未实现）
                        restored_img = self.deep_restoration(image)

                    restored_images.append(restored_img)
                    stats['count'] += 1

                    self.log_updated.emit(f"完成修复第 {i + 1} 张图像")

                except Exception as e:
                    self.log_updated.emit(f"修复第 {i + 1} 张图像失败: {str(e)}")
                    restored_images.append(None)

            if self.canceled:
                self.finished.emit(None)
                return

            # 计算统计信息
            end_time = time.time()
            elapsed = int(end_time - start_time)
            stats['time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            stats['avg_area'] = int(total_area / stats['count']) if stats['count'] > 0 else 0

            self.log_updated.emit(f"全部修复完成，共处理 {stats['count']}/{total} 张图像")
            self.progress_updated.emit(100, f"修复完成 ({stats['count']}/{total})")

            # 返回修复结果
            self.finished.emit((restored_images, stats))

        except Exception as e:
            self.log_updated.emit(f"修复过程出错: {str(e)}")
            self.finished.emit(None)

    def detect_damage_area(self, image):
        """检测图像损伤区域（模拟）"""
        # 实际应用中应替换为真实的损伤检测算法
        height, width = image.shape[:2]
        # 根据修复强度模拟损伤区域大小
        return int((height * width) * (0.05 + self.strength * 0.03))

    def quick_restoration(self, image):
        """快速修复算法：淡化图像污点"""
        # 转换为BGR格式进行处理
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. 先进行轻微降噪
        denoised = cv2.fastNlMeansDenoisingColored(
            bgr_img,
            None,
            self.denoise_level / 20,  # 降低降噪强度
            self.denoise_level / 20,
            7,
            21
        )

        # 2. 使用中值滤波淡化污点
        ksize = self.strength * 2 + 1  # 根据强度动态调整卷积核大小
        if ksize % 2 == 0:
            ksize += 1  # 确保是奇数

        restored = cv2.medianBlur(denoised, ksize)

        # 3. 轻微锐化，减少模糊感
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        restored = cv2.filter2D(restored, -1, kernel)

        # 转回RGB格式
        return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)

    def fine_restoration(self, image):
        """精细修复算法：完整修复数据集生成的图像"""
        # 转换为BGR格式进行处理
        bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. 转换为灰度图检测损伤区域
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # 2. 使用阈值和形态学操作识别损伤区域
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # 假设白色区域为损伤
        kernel = np.ones((3, 3), np.uint8)
        damage_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 3. 对损伤区域进行修复
        # 使用Telea算法进行图像修复
        restored = cv2.inpaint(
            bgr_img,
            damage_mask,
            self.strength,  # 修复半径
            cv2.INPAINT_TELEA
        )

        # 4. 应用双边滤波保留边缘和细节
        detail_preserved = cv2.bilateralFilter(
            restored,
            d=self.strength * 2 + 1,
            sigmaColor=75,
            sigmaSpace=75
        )

        # 5. 根据细节保留参数调整结果
        alpha = self.detail_level / 100.0
        final_result = cv2.addWeighted(restored, alpha, detail_preserved, 1 - alpha, 0)

        # 转回RGB格式
        return cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)

    def deep_restoration(self, image):
        """深度修复算法：预留接口，未实现"""
        # 深度修复算法实现将在这里添加
        # 目前直接返回原图的副本
        return image.copy()

    def cancel(self):
        """取消处理"""
        self.canceled = True

    def save_restored_image(self, image, original_path, index):
        """保存修复后的图像"""
        if image is None:
            return False

        try:
            # 构建输出路径
            filename = os.path.splitext(os.path.basename(original_path))[0]
            output_path = os.path.join(
                self.output_dir,
                f"{filename}_restored_{index}.{self.format}"
            )

            # 转换为BGR格式保存
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, bgr_image)
            return True

        except Exception as e:
            self.log_updated.emit(f"保存图像失败: {str(e)}")
            return False
