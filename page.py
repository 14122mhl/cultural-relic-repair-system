import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTextEdit, QPushButton, QFileDialog, QComboBox,
                             QProgressBar, QMessageBox)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal
import docx

# 百度语音语音合成API相关配置 - 请替换为自己的密钥
APP_ID = "120570017"
API_KEY = "eqg6zwcJ2yfUtpcClTWJ9KiS"
SECRET_KEY = "8JjVwAXNutxaHYxc2DHMIVfMksb5pxQm"

from aip import AipSpeech

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


class SpeechSynthesisThread(QThread):
    """语音合成线程，用于后台处理文本合成和播放"""
    signal_update_progress = pyqtSignal(int)
    signal_finish = pyqtSignal()

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.is_paused = False
        self.is_stopped = False
        self.player = QMediaPlayer()
        # PyQt5中没有QAudioOutput，使用setVolume直接控制音量

    def run(self):
        if not self.text:
            self.signal_finish.emit()
            return

        segments = self._split_text(self.text)
        total_segments = len(segments)
        for i, segment in enumerate(segments):
            if self.is_stopped:
                break
            while self.is_paused:
                time.sleep(0.1)
                if self.is_stopped:
                    break
            if self.is_stopped:
                break
            self.signal_update_progress.emit(int((i + 1) * 100 / total_segments))
            audio_data = self._synthesize_speech(segment)
            if audio_data:
                self._play_audio(audio_data)

        self.signal_finish.emit()

    def _split_text(self, text, max_length=1024):
        """将文本分割为适合百度API的长度"""
        segments = []
        current = ""
        for char in text:
            if len(current) + 1 <= max_length:
                current += char
            else:
                segments.append(current)
                current = char
        if current:
            segments.append(current)
        return segments

    def _synthesize_speech(self, text):
        """调用百度语音合成API生成语音"""
        result = client.synthesis(text, 'zh', 1, {
            'vol': 5, 'spd': 5, 'pit': 5, 'per': 4  # 音量、语速、音调、发音人
        })
        if not isinstance(result, dict):
            return result
        return None

    def _play_audio(self, audio_data):
        """播放语音数据"""
        temp_file = "temp_audio.mp3"
        try:
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_file)))
            self.player.setVolume(100)  # PyQt5中直接用setVolume设置音量（0-100）
            self.player.play()

            # 等待播放完成
            while self.player.state() == QMediaPlayer.PlayingState:
                if self.is_paused or self.is_stopped:
                    self.player.pause()
                    break
                time.sleep(0.1)
        finally:
            # 确保临时文件被删除
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def pause(self):
        self.is_paused = True
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()

    def resume(self):
        self.is_paused = False
        if self.player.state() == QMediaPlayer.PausedState:
            self.player.play()

    def stop(self):
        self.is_stopped = True
        self.is_paused = False
        if self.player.state() in [QMediaPlayer.PlayingState, QMediaPlayer.PausedState]:
            self.player.stop()


class SpeechSynthesisWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音合成")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # 选择输入方式
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入方式:"))
        self.input_combo = QComboBox()
        self.input_combo.addItems(["直接输入", "选择TXT文件", "选择Word文件"])
        self.input_combo.currentIndexChanged.connect(self._change_input_mode)
        input_layout.addWidget(self.input_combo)
        main_layout.addLayout(input_layout)

        # 文本输入区域
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("请输入要合成语音的文字...")
        main_layout.addWidget(self.text_edit)

        # 文件选择按钮
        self.file_button = QPushButton("选择文件")
        self.file_button.clicked.connect(self._select_file)
        self.file_button.setVisible(False)
        main_layout.addWidget(self.file_button)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.play_button = QPushButton("开始朗读")
        self.play_button.clicked.connect(self._start_playback)
        control_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("暂停朗读")
        self.pause_button.clicked.connect(self._pause_playback)
        self.pause_button.setEnabled(False)
        control_layout.addWidget(self.pause_button)

        self.resume_button = QPushButton("继续朗读")
        self.resume_button.clicked.connect(self._resume_playback)
        self.resume_button.setEnabled(False)
        control_layout.addWidget(self.resume_button)

        self.stop_button = QPushButton("停止朗读")
        self.stop_button.clicked.connect(self._stop_playback)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        main_layout.addLayout(control_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel("就绪")
        main_layout.addWidget(self.status_label)

        self.speech_thread = None

    def _change_input_mode(self, index):
        if index == 0:  # 直接输入
            self.text_edit.setVisible(True)
            self.file_button.setVisible(False)
        else:  # 选择文件
            self.text_edit.setVisible(False)
            self.file_button.setVisible(True)

    def _select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "文本文件 (*.txt);;Word文件 (*.docx)")
        if file_path:
            try:
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.text_edit.setText(f.read())
                elif file_path.endswith('.docx'):
                    doc = docx.Document(file_path)
                    text = '\n'.join([para.text for para in doc.paragraphs])
                    self.text_edit.setText(text)
                self.status_label.setText(f"已加载文件: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"文件读取失败: {str(e)}")

    def _get_text(self):
        return self.text_edit.toPlainText().strip()

    def _start_playback(self):
        text = self._get_text()
        if not text:
            QMessageBox.warning(self, "提示", "请输入或选择要朗读的内容")
            return

        self.speech_thread = SpeechSynthesisThread(text)
        self.speech_thread.signal_update_progress.connect(self.progress_bar.setValue)
        self.speech_thread.signal_finish.connect(self._playback_finished)
        self.speech_thread.start()

        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("正在朗读...")

    def _pause_playback(self):
        if self.speech_thread:
            self.speech_thread.pause()
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(True)
            self.status_label.setText("已暂停")

    def _resume_playback(self):
        if self.speech_thread:
            self.speech_thread.resume()
            self.pause_button.setEnabled(True)
            self.resume_button.setEnabled(False)
            self.status_label.setText("继续朗读...")

    def _stop_playback(self):
        if self.speech_thread:
            self.speech_thread.stop()
            self._reset_controls()

    def _playback_finished(self):
        self._reset_controls()
        self.status_label.setText("朗读完成")

    def _reset_controls(self):
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.speech_thread = None

    def closeEvent(self, event):
        if self.speech_thread:
            self.speech_thread.stop()
            self.speech_thread.wait()
        event.accept()


if __name__ == "__main__":
    # 安装依赖：pip install PyQt5 python-docx baidu-aip
    app = QApplication(sys.argv)
    window = SpeechSynthesisWindow()
    window.show()
    sys.exit(app.exec_())  # PyQt5中是exec_()