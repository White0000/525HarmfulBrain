import os
import sys
import numpy as np
import pandas as pd
import scipy.signal as sp
import pywt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog, QProgressBar, QComboBox, QLineEdit, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class DataLoadWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, object)
    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        try:
            df = pd.read_csv(self.path)
            self.progress_signal.emit(f"Loaded CSV: {self.path} with shape {df.shape}")
            self.finished_signal.emit(True, df)
        except Exception as e:
            self.progress_signal.emit(str(e))
            self.finished_signal.emit(False, None)

class FilterWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, object)
    def __init__(self, data, low, high, fs, order, notch, wavelet_denoise, wavelet_name, multi_channel, multi_thread):
        super().__init__()
        self.data = data
        self.low = low
        self.high = high
        self.fs = fs
        self.order = order
        self.notch = notch
        self.wavelet_denoise = wavelet_denoise
        self.wavelet_name = wavelet_name
        self.multi_channel = multi_channel
        self.multi_thread = multi_thread
    def notch_filter(self, x, freq=50.0, Q=30.0):
        b, a = sp.iirnotch(freq / (0.5 * self.fs), Q)
        return sp.filtfilt(b, a, x)
    def wavelet_denoise_func(self, x):
        c = pywt.wavedec(x, self.wavelet_name, mode="symmetric")
        for i in range(1, len(c)):
            c[i] = sp.medfilt(c[i], kernel_size=3)
        r = pywt.waverec(c, self.wavelet_name, mode="symmetric")
        return r[: len(x)]
    def filter_row(self, row):
        b, a = sp.butter(self.order, [self.low / (0.5 * self.fs), self.high / (0.5 * self.fs)], btype="bandpass")
        f = sp.filtfilt(b, a, row)
        if self.notch:
            f = self.notch_filter(f)
        if self.wavelet_denoise:
            f = self.wavelet_denoise_func(f)
        return f
    def run(self):
        try:
            if "label" in self.data.columns:
                labels = self.data["label"].values
                arr = self.data.drop("label", axis=1).values
            else:
                labels = None
                arr = self.data.values
            if self.multi_channel and arr.ndim == 2:
                rows = []
                for i, row in enumerate(arr):
                    row_2d = row.reshape(-1, order="C")
                    f = self.filter_row(row_2d)
                    rows.append(f)
                    if i % 50 == 0:
                        self.progress_signal.emit(f"Filtering row {i+1}/{arr.shape[0]}")
                df_filtered = pd.DataFrame(np.array(rows))
            else:
                if not self.multi_thread:
                    f_list = []
                    b, a = sp.butter(self.order, [self.low / (0.5 * self.fs), self.high / (0.5 * self.fs)], btype="bandpass")
                    for i, row in enumerate(arr):
                        x = sp.filtfilt(b, a, row)
                        if self.notch:
                            x = self.notch_filter(x)
                        if self.wavelet_denoise:
                            x = self.wavelet_denoise_func(x)
                        f_list.append(x)
                        if i % 50 == 0:
                            self.progress_signal.emit(f"Filtering row {i+1}/{arr.shape[0]}")
                    df_filtered = pd.DataFrame(np.array(f_list))
                else:
                    from concurrent.futures import ThreadPoolExecutor
                    bsz = 64
                    f_list = []
                    total = len(arr)
                    with ThreadPoolExecutor() as ex:
                        idx = 0
                        while idx < total:
                            batch = arr[idx : idx + bsz]
                            futures = []
                            for b_row in batch:
                                futures.append(ex.submit(self.filter_row, b_row))
                            for fut in futures:
                                f_list.append(fut.result())
                            idx += bsz
                            self.progress_signal.emit(f"Filtering rows up to {idx}/{total}")
                    df_filtered = pd.DataFrame(np.array(f_list))
            if labels is not None:
                df_filtered["label"] = labels
            self.finished_signal.emit(True, df_filtered)
        except Exception as e:
            self.progress_signal.emit(str(e))
            self.finished_signal.emit(False, None)

class SegmentWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, object)
    def __init__(self, data, step, overlap):
        super().__init__()
        self.data = data
        self.step = step
        self.overlap = overlap
    def run(self):
        try:
            if "label" in self.data.columns:
                label_arr = self.data["label"].values
                arr = self.data.drop("label", axis=1).values
            else:
                label_arr = None
                arr = self.data.values
            segments = []
            seg_labels = []
            for i, row in enumerate(arr):
                start = 0
                length = len(row)
                while start + self.step <= length:
                    seg = row[start : start + self.step]
                    segments.append(seg)
                    if label_arr is not None:
                        seg_labels.append(label_arr[i])
                    start += self.step - self.overlap
                if i % 50 == 0:
                    self.progress_signal.emit(f"Segmenting row {i+1}/{arr.shape[0]}")
            arr_segments = np.array(segments)
            df_segments = pd.DataFrame(arr_segments)
            if seg_labels:
                df_segments["label"] = seg_labels
            self.finished_signal.emit(True, df_segments)
        except Exception as e:
            self.progress_signal.emit(str(e))
            self.finished_signal.emit(False, None)

class SaveWorker(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)
    def __init__(self, data, path):
        super().__init__()
        self.data = data
        self.path = path
    def run(self):
        try:
            self.data.to_csv(self.path, index=False)
            self.progress_signal.emit(f"Saved to {self.path}")
            self.finished_signal.emit(True)
        except Exception as e:
            self.progress_signal.emit(str(e))
            self.finished_signal.emit(False)

class PreprocessWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preprocessing Pipeline")
        self.resize(900, 550)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.button_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Raw CSV")
        self.filter_btn = QPushButton("Apply Filter")
        self.segment_btn = QPushButton("Segment Data")
        self.save_btn = QPushButton("Save CSV")
        self.button_layout.addWidget(self.load_btn)
        self.button_layout.addWidget(self.filter_btn)
        self.button_layout.addWidget(self.segment_btn)
        self.button_layout.addWidget(self.save_btn)
        self.layout.addLayout(self.button_layout)
        self.param_layout = QHBoxLayout()
        self.low_label = QLabel("Low Hz:")
        self.high_label = QLabel("High Hz:")
        self.low_edit = QLineEdit("0.5")
        self.high_edit = QLineEdit("30")
        self.sr_label = QLabel("Sample Rate:")
        self.sr_edit = QLineEdit("100")
        self.order_label = QLabel("Order:")
        self.order_edit = QLineEdit("4")
        self.notch_check = QCheckBox("Notch 50Hz")
        self.wavelet_check = QCheckBox("Wavelet Denoise")
        self.wavelet_label = QLabel("Wavelet:")
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(["db4", "db1", "coif1", "sym2"])
        self.multi_ch_check = QCheckBox("Multi-Channel Filter")
        self.mt_check = QCheckBox("Multi-Thread")
        self.segment_label = QLabel("Segment Size:")
        self.segment_edit = QLineEdit("100")
        self.overlap_label = QLabel("Overlap:")
        self.overlap_edit = QLineEdit("0")
        self.param_layout.addWidget(self.low_label)
        self.param_layout.addWidget(self.low_edit)
        self.param_layout.addWidget(self.high_label)
        self.param_layout.addWidget(self.high_edit)
        self.param_layout.addWidget(self.sr_label)
        self.param_layout.addWidget(self.sr_edit)
        self.param_layout.addWidget(self.order_label)
        self.param_layout.addWidget(self.order_edit)
        self.param_layout.addWidget(self.notch_check)
        self.param_layout.addWidget(self.wavelet_check)
        self.param_layout.addWidget(self.wavelet_label)
        self.param_layout.addWidget(self.wavelet_combo)
        self.param_layout.addWidget(self.multi_ch_check)
        self.param_layout.addWidget(self.mt_check)
        self.param_layout.addWidget(self.segment_label)
        self.param_layout.addWidget(self.segment_edit)
        self.param_layout.addWidget(self.overlap_label)
        self.param_layout.addWidget(self.overlap_edit)
        self.layout.addLayout(self.param_layout)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(self.log_area)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        self.layout.addWidget(self.progress_bar)
        self.raw_data = None
        self.filtered_data = None
        self.segmented_data = None
        self.load_btn.clicked.connect(self.load_csv)
        self.filter_btn.clicked.connect(self.apply_filter)
        self.segment_btn.clicked.connect(self.segment_data)
        self.save_btn.clicked.connect(self.save_csv)

    def log(self, msg):
        self.log_area.append(msg)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Raw CSV", "", "CSV Files (*.csv)")
        if path:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.show()
            self.load_btn.setEnabled(False)
            self.filter_btn.setEnabled(False)
            self.segment_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.load_thread = DataLoadWorker(path)
            self.load_thread.progress_signal.connect(self.log)
            self.load_thread.finished_signal.connect(self.load_finished)
            self.load_thread.start()

    def load_finished(self, status, df):
        self.progress_bar.hide()
        self.load_btn.setEnabled(True)
        self.filter_btn.setEnabled(True)
        self.segment_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        if status:
            self.raw_data = df
            self.filtered_data = None
            self.segmented_data = None
        else:
            QMessageBox.critical(self, "Error", "Failed to load CSV.")

    def apply_filter(self):
        if self.raw_data is None:
            self.log("No raw data.")
            return
        try:
            low = float(self.low_edit.text())
            high = float(self.high_edit.text())
            sr = float(self.sr_edit.text())
            order = int(self.order_edit.text())
            notch = self.notch_check.isChecked()
            wavelet_denoise = self.wavelet_check.isChecked()
            wavelet_name = self.wavelet_combo.currentText()
            multi_ch = self.multi_ch_check.isChecked()
            mt = self.mt_check.isChecked()
            self.progress_bar.setRange(0, 0)
            self.progress_bar.show()
            self.load_btn.setEnabled(False)
            self.filter_btn.setEnabled(False)
            self.segment_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.filter_thread = FilterWorker(self.raw_data, low, high, sr, order, notch, wavelet_denoise, wavelet_name, multi_ch, mt)
            self.filter_thread.progress_signal.connect(self.log)
            self.filter_thread.finished_signal.connect(self.filter_finished)
            self.filter_thread.start()
        except Exception as e:
            self.log(str(e))

    def filter_finished(self, status, df):
        self.progress_bar.hide()
        self.load_btn.setEnabled(True)
        self.filter_btn.setEnabled(True)
        self.segment_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        if status:
            self.filtered_data = df
            self.segmented_data = None
        else:
            QMessageBox.critical(self, "Error", "Filter process failed.")

    def segment_data(self):
        if self.filtered_data is None:
            self.log("No filtered data.")
            return
        try:
            step = int(self.segment_edit.text())
            overlap = int(self.overlap_edit.text())
            self.progress_bar.setRange(0, 0)
            self.progress_bar.show()
            self.load_btn.setEnabled(False)
            self.filter_btn.setEnabled(False)
            self.segment_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.segment_thread = SegmentWorker(self.filtered_data, step, overlap)
            self.segment_thread.progress_signal.connect(self.log)
            self.segment_thread.finished_signal.connect(self.segment_finished)
            self.segment_thread.start()
        except Exception as e:
            self.log(str(e))

    def segment_finished(self, status, df):
        self.progress_bar.hide()
        self.load_btn.setEnabled(True)
        self.filter_btn.setEnabled(True)
        self.segment_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        if status:
            self.segmented_data = df
        else:
            QMessageBox.critical(self, "Error", "Segmentation process failed.")

    def save_csv(self):
        if self.segmented_data is None:
            self.log("No segmented data.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Processed CSV", "", "CSV Files (*.csv)")
        if path:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.show()
            self.load_btn.setEnabled(False)
            self.filter_btn.setEnabled(False)
            self.segment_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.save_thread = SaveWorker(self.segmented_data, path)
            self.save_thread.progress_signal.connect(self.log)
            self.save_thread.finished_signal.connect(self.save_finished)
            self.save_thread.start()

    def save_finished(self, status):
        self.progress_bar.hide()
        self.load_btn.setEnabled(True)
        self.filter_btn.setEnabled(True)
        self.segment_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        if not status:
            QMessageBox.critical(self, "Error", "Save failed.")

def main():
    app = QApplication(sys.argv)
    w = PreprocessWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
