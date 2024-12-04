import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QPushButton, QCheckBox, QFileDialog, QTextEdit,
                           QProgressBar, QLabel, QMessageBox, QGroupBox, QHBoxLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from reconstruction import ReconstructionWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Reconstruction GUI")
        self.setGeometry(100, 100, 800, 600)
        self.worker = None
        self.thread = None
        self.selected_folder = None
        self.output_folder = None
        self.initUI()

    def initUI(self):
        # 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 文件夹选择部分
        input_group = QGroupBox("Input Settings")
        input_layout = QVBoxLayout()
        
        # 输入文件夹选择
        input_folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Input Folder: None")
        select_folder_btn = QPushButton("Select Image Folder")
        select_folder_btn.clicked.connect(self.selectFolder)
        input_folder_layout.addWidget(self.folder_label)
        input_folder_layout.addWidget(select_folder_btn)
        input_layout.addLayout(input_folder_layout)
        
        # 输出文件夹选择
        output_folder_layout = QHBoxLayout()
        self.output_label = QLabel("Output Folder: Default")
        select_output_btn = QPushButton("Select Output Folder")
        select_output_btn.clicked.connect(self.selectOutputFolder)
        output_folder_layout.addWidget(self.output_label)
        output_folder_layout.addWidget(select_output_btn)
        input_layout.addLayout(output_folder_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # 配置选项
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        self.texture_checkbox = QCheckBox("Enable Texture Reconstruction")
        self.deeplearning_checkbox = QCheckBox("Enable Deep Learning (MiDaS)")
        config_layout.addWidget(self.texture_checkbox)
        config_layout.addWidget(self.deeplearning_checkbox)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # 控制按钮
        control_group = QGroupBox("Control")
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Reconstruction")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.clicked.connect(self.startReconstruction)
        self.stop_btn.clicked.connect(self.stopReconstruction)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 进度条
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # 日志区域
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

    def selectOutputFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_label.setText(f"Output Folder: {folder}")

    def selectFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.selected_folder = folder
            self.folder_label.setText(f"Input Folder: {folder}")
            # 设置默认输出目录
            default_output = os.path.join(folder, "reconstruction_output")
            self.output_folder = default_output
            self.output_label.setText(f"Output Folder: {default_output}")
            self.validateInputs()

    def validateInputs(self):
        # 检查是否选择了文件夹并包含支持的图像
        if self.selected_folder:
            image_files = [f for f in os.listdir(self.selected_folder)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.start_btn.setEnabled(len(image_files) > 0)
        else:
            self.start_btn.setEnabled(False)

    def startReconstruction(self):
        if not self.selected_folder:
            QMessageBox.warning(self, "Warning", "Please select an image folder first!")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 创建工作线程
        self.thread = QThread()
        self.worker = ReconstructionWorker(
            input_folder=self.selected_folder,
            output_folder=self.output_folder,
            use_texture=self.texture_checkbox.isChecked(),
            use_deeplearning=self.deeplearning_checkbox.isChecked()
        )
        self.worker.moveToThread(self.thread)

        # 连接信号
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.updateProgress)
        self.worker.log.connect(self.updateLog)
        self.worker.error.connect(self.handleError)

        # 启动线程
        self.thread.start()

    def stopReconstruction(self):
        if self.worker:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.updateLog("Stopping reconstruction process...")

    def updateProgress(self, value):
        self.progress_bar.setValue(value)

    def updateLog(self, message):
        self.log_text.append(message)

    def handleError(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        if self.worker and self.thread and self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
