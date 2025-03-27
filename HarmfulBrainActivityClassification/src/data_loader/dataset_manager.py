import os
import sys
import random
import numpy as np
import pandas as pd
import scipy.signal as sp
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QComboBox, QCheckBox, QProgressBar, QMessageBox, QInputDialog, QDialog, QListWidget, QListWidgetItem, QAbstractItemView, QDialogButtonBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sklearn.preprocessing import LabelEncoder

class BandpassFilterTransform:
    def __init__(self, low, high, fs, order=4):
        self.low = low
        self.high = high
        self.fs = fs
        self.order = order
        nyq = 0.5 * fs
        self.b, self.a = sp.butter(self.order, [low / nyq, high / nyq], btype="bandpass")
    def __call__(self, x):
        return sp.filtfilt(self.b, self.a, x).astype(np.float32)

class RandomNoiseTransform:
    def __init__(self, noise_factor=0.01):
        self.noise_factor = noise_factor
    def __call__(self, x):
        return (x + self.noise_factor * np.random.randn(*x.shape)).astype(np.float32)

class CombinedTransform:
    def __init__(self, transforms=None):
        self.transforms = transforms if transforms else []
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class EEGDataset(Dataset):
    def __init__(self, df, transform=None, label_col="label", auto_encode=True):
        self.transform = transform
        if label_col not in df.columns:
            raise ValueError(f"No column named '{label_col}' found.")
        lab = df[label_col].values
        if auto_encode:
            numeric = True
            try:
                _ = lab.astype(np.int64)
            except:
                numeric = False
            if not numeric:
                le = LabelEncoder()
                df[label_col] = le.fit_transform(df[label_col].astype(str))
        self.x_data = df.drop(label_col, axis=1).values.astype(np.float32)
        self.y_data = df[label_col].values.astype(np.int64)
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        if self.transform:
            x = self.transform(x)
        return torch.from_numpy(x), torch.tensor(y)

class DataManager:
    def __init__(self, csv_path=None, batch_size=32, shuffle=True, transform=None, label_col="label"):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.label_col = label_col
        self.df = None
        self.dataset = None
    def load_csv(self):
        if self.csv_path and os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
    def unify_votes_to_label(self, cols, label_map=None):
        if self.df is None:
            return
        if label_map is None:
            label_map = {"Seizure": 0, "GPD": 1, "LRDA": 2, "GRDA": 3, "Other": 4}
        if "expert_consensus" in self.df.columns:
            if "expert_consensus" not in cols:
                pass
        else:
            pass
        arr = []
        for i, r in self.df.iterrows():
            if "expert_consensus" in cols:
                arr.append(r["expert_consensus"])
            else:
                votes = {}
                for c in cols:
                    if r[c] > 0:
                        votes[c] = r[c]
                if len(votes) == 0:
                    arr.append("Other")
                else:
                    mx = max(votes, key=votes.get)
                    if mx.lower().startswith("seizure"):
                        arr.append("Seizure")
                    elif mx.lower().startswith("lpd"):
                        arr.append("GPD")
                    elif mx.lower().startswith("gpd"):
                        arr.append("GPD")
                    elif mx.lower().startswith("lrda"):
                        arr.append("LRDA")
                    elif mx.lower().startswith("grda"):
                        arr.append("GRDA")
                    else:
                        arr.append("Other")
        self.df["label"] = arr
        le = LabelEncoder()
        self.df["label"] = le.fit_transform(self.df["label"].astype(str))
    def select_features(self, features):
        if self.df is None:
            return
        all_cols = list(self.df.columns)
        keep = []
        for c in features:
            if c in all_cols:
                keep.append(c)
        if "label" in self.df.columns and "label" not in keep:
            keep.append("label")
        self.df = self.df[keep]
    def create_dataset(self):
        if self.df is None:
            return
        if self.label_col not in self.df.columns:
            raise ValueError("Label column not found.")
        self.dataset = EEGDataset(self.df, transform=self.transform, label_col=self.label_col, auto_encode=True)
    def get_weights(self):
        labels = self.dataset.y_data
        counts = np.bincount(labels)
        total = sum(counts)
        w = [total / c if c > 0 else 0 for c in counts]
        return [w[int(lbl)] for lbl in labels]
    def create_loader(self, balanced=False):
        if not self.dataset:
            return None
        if balanced:
            sampler = WeightedRandomSampler(self.get_weights(), len(self.dataset))
            return DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler)
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    def dataset_info(self):
        if not self.dataset:
            return {}
        x_len = len(self.dataset.x_data)
        features = self.dataset.x_data.shape[1]
        labels = self.dataset.y_data
        counts = np.bincount(labels)
        return {"samples": x_len, "features": features, "distribution": counts}

class FeatureSelectionDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Feature Columns")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for c in columns:
            item = QListWidgetItem(c)
            self.list_widget.addItem(item)
        self.layout.addWidget(self.list_widget)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
    def get_selected_features(self):
        selected = []
        for item in self.list_widget.selectedItems():
            selected.append(item.text())
        return selected

class LoadCSVThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
    def run(self):
        try:
            self.manager.load_csv()
            self.progress_signal.emit("CSV loaded.")
            self.finished_signal.emit(True)
        except Exception as e:
            self.progress_signal.emit(str(e))
            self.finished_signal.emit(False)

class DatasetManagerWindow(QWidget):
    def __init__(self, manager=None, parent=None):
        super().__init__(parent)
        self.manager = manager if manager else DataManager()
        self.transform_pipeline = CombinedTransform()
        self.setWindowTitle("EEG Dataset Manager")
        self.layout = QVBoxLayout()
        self.top_layout = QHBoxLayout()
        self.load_button = QPushButton("Load CSV")
        self.advanced_label_button = QPushButton("Advanced Labeling")
        self.select_features_button = QPushButton("Select Features")
        self.create_loader_button = QPushButton("Create DataLoader")
        self.info_button = QPushButton("Show Dataset Info")
        self.layout.addLayout(self.top_layout)
        self.top_layout.addWidget(self.load_button)
        self.top_layout.addWidget(self.advanced_label_button)
        self.top_layout.addWidget(self.select_features_button)
        self.top_layout.addWidget(self.create_loader_button)
        self.top_layout.addWidget(self.info_button)
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.batch_box = QComboBox()
        self.batch_box.addItems(["8", "16", "32", "64", "128"])
        self.filter_box = QCheckBox("Apply Bandpass Filter (1-30Hz)")
        self.noise_box = QCheckBox("Add Random Noise")
        self.balance_box = QCheckBox("Use Balanced Sampler")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)
        self.progress.hide()
        self.layout.addWidget(QLabel("Batch Size:"))
        self.layout.addWidget(self.batch_box)
        self.layout.addWidget(self.filter_box)
        self.layout.addWidget(self.noise_box)
        self.layout.addWidget(self.balance_box)
        self.layout.addWidget(self.text_area)
        self.layout.addWidget(self.progress)
        self.setLayout(self.layout)
        self.resize(900, 500)
        self.load_button.clicked.connect(self.load_csv)
        self.advanced_label_button.clicked.connect(self.advanced_labeling)
        self.select_features_button.clicked.connect(self.select_features)
        self.create_loader_button.clicked.connect(self.create_loader)
        self.info_button.clicked.connect(self.show_info)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if path:
            self.text_area.append(f"Loading {path}...")
            self.manager.csv_path = path
            self.load_thread = LoadCSVThread(self.manager)
            self.load_thread.progress_signal.connect(self.handle_progress)
            self.load_thread.finished_signal.connect(self.load_finish)
            self.progress.show()
            self.load_button.setEnabled(False)
            self.advanced_label_button.setEnabled(False)
            self.select_features_button.setEnabled(False)
            self.create_loader_button.setEnabled(False)
            self.info_button.setEnabled(False)
            self.load_thread.start()

    def advanced_labeling(self):
        if self.manager.df is None:
            self.text_area.append("No data loaded.")
            return
        columns = list(self.manager.df.columns)
        choices = ["Use expert_consensus", "Merge votes (seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote)"]
        c, ok = QInputDialog.getItem(self, "Advanced Labeling", "Choose labeling strategy:", choices, 0, False)
        if ok and c:
            if "expert_consensus" in columns and c.startswith("Use expert"):
                self.manager.label_col = "label"
                self.manager.unify_votes_to_label(["expert_consensus"])
                self.text_area.append("Set label based on expert_consensus.")
            else:
                vote_cols = []
                for v in ["seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote"]:
                    if v in columns:
                        vote_cols.append(v)
                self.manager.label_col = "label"
                self.manager.unify_votes_to_label(vote_cols)
                self.text_area.append("Merged votes into label.")

    def select_features(self):
        if self.manager.df is None:
            self.text_area.append("No data loaded.")
            return
        columns = list(self.manager.df.columns)
        dialog = FeatureSelectionDialog(columns, self)
        if dialog.exec_() == QDialog.Accepted:
            selected = dialog.get_selected_features()
            self.manager.select_features(selected)
            self.text_area.append(f"Selected features: {selected}")

    def handle_progress(self, msg):
        self.text_area.append(msg)

    def load_finish(self, status):
        self.progress.hide()
        self.load_button.setEnabled(True)
        self.advanced_label_button.setEnabled(True)
        self.select_features_button.setEnabled(True)
        self.create_loader_button.setEnabled(True)
        self.info_button.setEnabled(True)
        if not status:
            QMessageBox.critical(self, "Error", "Failed to load dataset.")
        else:
            self.text_area.append("CSV loaded. You may now do advanced labeling or select features.")

    def create_loader(self):
        if self.manager.df is None:
            self.text_area.append("No data to create loader.")
            return
        b = int(self.batch_box.currentText())
        self.manager.batch_size = b
        transforms_list = []
        if self.filter_box.isChecked():
            transforms_list.append(BandpassFilterTransform(1, 30, 128, order=4))
        if self.noise_box.isChecked():
            transforms_list.append(RandomNoiseTransform(0.02))
        self.transform_pipeline.transforms = transforms_list
        self.manager.transform = self.transform_pipeline
        try:
            self.manager.create_dataset()
            loader = self.manager.create_loader(balanced=self.balance_box.isChecked())
            if loader:
                it = iter(loader)
                batch_x, batch_y = next(it)
                self.text_area.append(f"Loader created. First batch shapes: X={batch_x.shape}, Y={batch_y.shape}")
            else:
                self.text_area.append("Failed to create loader.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def show_info(self):
        info = self.manager.dataset_info()
        if not info:
            self.text_area.append("No dataset loaded.")
            return
        s = f"Samples: {info['samples']} | Features: {info['features']} | Distribution: {list(info['distribution'])}"
        self.text_area.append(s)

def main():
    app = QApplication(sys.argv)
    w = DatasetManagerWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
