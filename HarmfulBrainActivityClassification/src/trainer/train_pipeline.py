import sys
import os
import random
import numpy as np
import pandas as pd
import scipy.signal as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel,
    QHBoxLayout, QProgressBar, QComboBox, QLineEdit, QFileDialog, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from sklearn.preprocessing import LabelEncoder

class EEGDataset(Dataset):
    def __init__(self, df, transform=None, label_col="label", auto_encode=True):
        self.transform = transform
        if label_col not in df.columns:
            raise ValueError(f"No column named '{label_col}' found.")
        lab = df[label_col].values
        numeric = True
        try:
            _ = lab.astype(np.int64)
        except:
            numeric = False
        if auto_encode and (not numeric):
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

    def create_dataset(self):
        if self.df is None:
            return
        if self.label_col not in self.df.columns:
            raise ValueError("Label column not found.")
        self.dataset = EEGDataset(self.df, transform=self.transform, label_col=self.label_col, auto_encode=True)

    def create_loader(self, balanced=False):
        if not self.dataset:
            return None
        if balanced:
            labels = self.dataset.y_data
            counts = np.bincount(labels)
            total = sum(counts)
            w = [total/c if c>0 else 0 for c in counts]
            sample_weights = [w[lbl] for lbl in labels]
            sampler = WeightedRandomSampler(sample_weights, len(self.dataset))
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

class TrainerWorker(QThread):
    epoch_signal = pyqtSignal(int, int, float, float)
    finished_signal = pyqtSignal(bool, str)

    def __init__(
        self,
        model,
        loader,
        device,
        epochs=5,
        optimizer_type="adam",
        lr=1e-3,
        wd=1e-4,
        momentum=0.9,
        sched_type=None,
        sched_args=None
    ):
        super().__init__()
        self.model = model
        self.loader = loader
        self.device = device
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.sched_type = sched_type
        self.sched_args = sched_args if sched_args else {}
        self.stop_flag = False

    def run(self):
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()

        if self.optimizer_type.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)

        if self.sched_type == "step":
            step_size = self.sched_args.get("step_size", 10)
            gamma = self.sched_args.get("gamma", 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.sched_type == "cosine":
            tmax = self.sched_args.get("T_max", 50)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        else:
            scheduler = None

        for epoch in range(self.epochs):
            if self.stop_flag:
                self.finished_signal.emit(False, "Training Stopped by User.")
                return

            self.model.train()
            total_loss = 0.0
            total_samples = 0
            correct = 0

            for x_batch, y_batch in self.loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    bsz = x_batch.size(0)
                    correct += (preds == y_batch).sum().item()
                    total_samples += bsz
                    total_loss += loss.item() * bsz

            avg_loss = total_loss / total_samples if total_samples else 0.0
            acc = correct / total_samples if total_samples else 0.0
            self.epoch_signal.emit(epoch + 1, self.epochs, avg_loss, acc)

            if scheduler:
                scheduler.step()

        if not self.stop_flag:
            self.finished_signal.emit(True, "Training Completed Successfully.")
        else:
            self.finished_signal.emit(False, "Training Stopped by User.")

    def stop(self):
        self.stop_flag = True

class TrainPipelineWindow(QWidget):
    def __init__(self, manager=None, parent=None):
        super().__init__(parent)
        if manager is None:
            manager = DataManager()
        self.data_manager = manager

        self.setWindowTitle("Training Pipeline")
        self.resize(900, 600)
        self.model = None
        self.trainer_thread = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        top_layout = QHBoxLayout()
        self.load_data_btn = QPushButton("Load Dataset from Manager")
        self.train_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setEnabled(False)
        top_layout.addWidget(self.load_data_btn)
        top_layout.addWidget(self.train_btn)
        top_layout.addWidget(self.stop_btn)

        device_layout = QHBoxLayout()
        self.device_label = QLabel("Device:")
        self.device_box = QComboBox()
        self.device_box.addItem("cpu")
        if torch.cuda.is_available():
            self.device_box.addItem("cuda")

        self.model_label = QLabel("Model:")
        self.model_box = QComboBox()
        self.model_box.addItems(["Linear", "CNN", "LSTM", "Transformer"])

        self.lr_label = QLabel("LR:")
        self.lr_edit = QLineEdit("0.001")
        self.epochs_label = QLabel("Epochs:")
        self.epochs_edit = QLineEdit("5")

        self.opt_label = QLabel("Optimizer:")
        self.opt_box = QComboBox()
        self.opt_box.addItems(["adam", "sgd"])

        self.wd_label = QLabel("WeightDecay:")
        self.wd_edit = QLineEdit("0.0001")

        self.mom_label = QLabel("Momentum:")
        self.mom_edit = QLineEdit("0.9")

        self.scheduler_label = QLabel("Scheduler:")
        self.scheduler_box = QComboBox()
        self.scheduler_box.addItems(["none", "step", "cosine"])

        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.device_box)
        device_layout.addWidget(self.model_label)
        device_layout.addWidget(self.model_box)
        device_layout.addWidget(self.lr_label)
        device_layout.addWidget(self.lr_edit)
        device_layout.addWidget(self.epochs_label)
        device_layout.addWidget(self.epochs_edit)
        device_layout.addWidget(self.opt_label)
        device_layout.addWidget(self.opt_box)
        device_layout.addWidget(self.wd_label)
        device_layout.addWidget(self.wd_edit)
        device_layout.addWidget(self.mom_label)
        device_layout.addWidget(self.mom_edit)
        device_layout.addWidget(self.scheduler_label)
        device_layout.addWidget(self.scheduler_box)

        self.status_label = QLabel("Status: Idle")
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.layout.addLayout(top_layout)
        self.layout.addLayout(device_layout)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.log_area)

        self.load_data_btn.clicked.connect(self.load_dataset_from_manager)
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)

    def load_dataset_from_manager(self):
        self.log_area.append("Using dataset loaded by DatasetManagerWindow (if any).")
        if not self.data_manager.dataset:
            self.log_area.append("No valid dataset found inside data_manager.")
            return
        info = self.data_manager.dataset_info()
        if info:
            dist_str = ", ".join(str(c) for c in info["distribution"])
            self.log_area.append(f"Samples={info['samples']} | Features={info['features']} | Dist={dist_str}")

    def create_model(self, in_dim, out_dim):
        choice = self.model_box.currentText().lower()
        if choice == "linear":
            return nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(32, out_dim)
            )
        elif choice == "cnn":
            from src.models.cnn_model import SimpleCNN
            return SimpleCNN(in_channels=1, num_classes=out_dim)
        elif choice == "lstm":
            from src.models.rnn_model import LSTMClassifier
            return LSTMClassifier(
                input_size=in_dim,
                hidden_size=64,
                num_layers=2,
                num_classes=out_dim,
                bidirectional=True,
                dropout=0.2
            )
        elif choice == "transformer":
            from src.models.transformer_model import EEGTransformer
            return EEGTransformer(
                input_dim=in_dim,
                d_model=64,
                nhead=4,
                num_layers=2,
                num_classes=out_dim,
                dim_feedforward=256,
                dropout=0.2
            )
        else:
            return None

    def start_training(self):
        if not self.data_manager.dataset:
            self.log_area.append("Cannot train: no dataset loaded.")
            return

        self.status_label.setText("Status: Training...")
        self.progress_bar.setValue(0)
        self.log_area.append("Creating DataLoader...")

        loader = self.data_manager.create_loader(balanced=False)
        if not loader:
            self.log_area.append("Failed to create DataLoader. Check dataset.")
            return

        in_dim = self.data_manager.dataset.x_data.shape[1]
        out_dim = len(set(self.data_manager.dataset.y_data))

        net = self.create_model(in_dim, out_dim)
        if not net:
            self.log_area.append("Invalid model choice.")
            return

        device = torch.device(self.device_box.currentText())

        try:
            epochs = int(self.epochs_edit.text())
        except:
            epochs = 5
        try:
            lr = float(self.lr_edit.text())
        except:
            lr = 1e-3

        opt_type = self.opt_box.currentText()
        try:
            wd = float(self.wd_edit.text())
        except:
            wd = 1e-4
        try:
            momentum = float(self.mom_edit.text())
        except:
            momentum = 0.9

        sched_type = self.scheduler_box.currentText().lower()

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.trainer_thread = TrainerWorker(
            net,
            loader,
            device,
            epochs=epochs,
            optimizer_type=opt_type,
            lr=lr,
            wd=wd,
            momentum=momentum,
            sched_type=sched_type,
            sched_args={"step_size":10, "gamma":0.1, "T_max":50}
        )
        self.trainer_thread.epoch_signal.connect(self.handle_epoch_result)
        self.trainer_thread.finished_signal.connect(self.handle_train_finish)
        self.trainer_thread.start()

    def stop_training(self):
        if self.trainer_thread:
            self.trainer_thread.stop()
            self.log_area.append("Stop signal sent.")

    def handle_epoch_result(self, epoch_idx, total_epochs, avg_loss, acc):
        self.log_area.append(f"Epoch {epoch_idx}/{total_epochs} | Loss={avg_loss:.4f} | ACC={acc:.4f}")
        progress = int((epoch_idx / total_epochs) * 100)
        self.progress_bar.setValue(progress)

    def handle_train_finish(self, success, message):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Status: {message}")
        self.log_area.append(message)
        if success:
            self.progress_bar.setValue(100)
            # 提示是否保存整个模型 (而非 state_dict)
            ret = QMessageBox.question(
                self,
                "Save Model",
                "Training finished successfully. Do you want to save the ENTIRE model (nn.Module)?",
                QMessageBox.Yes | QMessageBox.No
            )
            if ret == QMessageBox.Yes:
                path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "PyTorch Model (*.pth *.pt)")
                if path:
                    try:
                        # 直接保存完整模型对象
                        torch.save(self.trainer_thread.model, path)
                        self.log_area.append(f"Full model saved to {path}")
                    except Exception as e:
                        QMessageBox.critical(self, "Error", str(e))
        else:
            self.log_area.append("Training was stopped, so no model saved.")
        self.trainer_thread = None


def main():
    app = QApplication(sys.argv)
    w = TrainPipelineWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
