import sys
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QFileDialog, QComboBox, QProgressBar, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from src.data_loader.dataset_manager import DataManager

class InferenceWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, float, list, list, dict)
    def __init__(self, model, loader, device, compute_auc=False, multi_label=False):
        super().__init__()
        self.model = model
        self.loader = loader
        self.device = device
        self.compute_auc = compute_auc
        self.multi_label = multi_label
    def run(self):
        try:
            self.model.to(self.device)
            self.model.eval()
            total = 0
            correct = 0
            all_preds = []
            all_labels = []
            all_outputs = []
            for i, (x_batch, y_batch) in enumerate(self.loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                with torch.no_grad():
                    logits = self.model(x_batch)
                if self.multi_label:
                    preds = (torch.sigmoid(logits) > 0.5).int()
                    correct += (preds == y_batch).all(dim=1).sum().item()
                else:
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y_batch).sum().item()
                total += x_batch.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_outputs.append(logits.cpu())
                p = int(100 * (i + 1) / len(self.loader))
                self.progress_signal.emit(p)
            acc = correct / total if total else 0
            metrics_dict = {}
            if self.compute_auc and not self.multi_label:
                import torch.nn.functional as F
                outputs_cat = torch.cat(all_outputs, dim=0)
                probs = F.softmax(outputs_cat, dim=1).numpy()
                try:
                    if probs.shape[1] == 2:
                        from sklearn.metrics import roc_auc_score
                        labels_np = np.array(all_labels)
                        auc_val = roc_auc_score(labels_np, probs[:, 1])
                        metrics_dict["auc"] = auc_val
                    else:
                        from sklearn.metrics import roc_auc_score
                        labels_np = np.array(all_labels)
                        auc_val = roc_auc_score(labels_np, probs, average="macro", multi_class="ovr")
                        metrics_dict["auc"] = auc_val
                except:
                    metrics_dict["auc"] = None
            self.finished_signal.emit(True, acc, all_preds, all_labels, metrics_dict)
        except Exception as e:
            self.log_signal.emit(str(e))
            self.finished_signal.emit(False, 0.0, [], [], {})

class InferenceWindow(QWidget):
    def __init__(self, manager=None, parent=None):
        super().__init__(parent)
        if manager is None:
            manager = DataManager()
        self.manager = manager
        self.setWindowTitle("Inference Pipeline")
        self.resize(850, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.info_label = QLabel("Status: Idle")
        self.load_model_btn = QPushButton("Load Trained Model")
        self.load_csv_btn = QPushButton("Load Inference CSV")
        self.run_inference_btn = QPushButton("Run Inference")
        self.save_pred_btn = QPushButton("Save Predictions to CSV")
        self.h_layout = QHBoxLayout()
        self.h_layout.addWidget(self.load_model_btn)
        self.h_layout.addWidget(self.load_csv_btn)
        self.h_layout.addWidget(self.run_inference_btn)
        self.h_layout.addWidget(self.save_pred_btn)
        self.layout.addWidget(self.info_label)
        self.layout.addLayout(self.h_layout)
        d_layout = QHBoxLayout()
        self.device_label = QLabel("Select Device:")
        self.device_box = QComboBox()
        self.device_box.addItem("cpu")
        if torch.cuda.is_available():
            self.device_box.addItem("cuda")
        self.auc_check = QCheckBox("Compute AUC")
        self.multi_label_check = QCheckBox("Multi-label")
        d_layout.addWidget(self.device_label)
        d_layout.addWidget(self.device_box)
        d_layout.addWidget(self.auc_check)
        d_layout.addWidget(self.multi_label_check)
        self.layout.addLayout(d_layout)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.layout.addWidget(self.progress)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(self.log_area)
        self.model = None
        self.preds = []
        self.labels = []
        self.extra_metrics = {}
        self.load_model_btn.clicked.connect(self.load_trained_model)
        self.load_csv_btn.clicked.connect(self.load_inference_csv)
        self.run_inference_btn.clicked.connect(self.run_inference)
        self.save_pred_btn.clicked.connect(self.save_predictions)

    def load_trained_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Weights (*.pth *.pt)")
        if path:
            try:
                obj = torch.load(path, map_location="cpu")
                if isinstance(obj, nn.Module):
                    self.model = obj
                    self.log_area.append(f"Model loaded: {os.path.basename(path)}")
                    self.info_label.setText("Status: Model Ready")
                else:
                    self.model = None
                    self.log_area.append("Invalid model format. Please select a valid nn.Module file.")
            except Exception as e:
                self.log_area.append(str(e))
                self.model = None

    def load_inference_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File for Inference", "", "CSV Files (*.csv)")
        if path:
            self.manager.csv_path = path
            self.manager.load_dataset()
            if self.manager.dataset:
                info = self.manager.dataset_info()
                self.info_label.setText("Status: Data Ready")
                if info:
                    dist = ", ".join(str(c) for c in info["distribution"])
                    self.log_area.append(f"Inference data loaded: {os.path.basename(path)}")
                    self.log_area.append(f"Samples={info['samples']} | Features={info['features']} | Distribution=[{dist}]")
                else:
                    self.log_area.append("Dataset info unavailable.")
            else:
                QMessageBox.warning(self, "Warning", "Failed to load dataset. Check CSV format.")
                self.log_area.append("Dataset load failed.")

    def run_inference(self):
        if not self.model or not isinstance(self.model, nn.Module):
            self.log_area.append("No valid model available for inference.")
            return
        if not self.manager.dataset:
            self.log_area.append("No dataset loaded for inference.")
            return
        loader = self.manager.create_loader(balanced=False)
        if not loader:
            self.log_area.append("Failed to create DataLoader. Check dataset.")
            return
        self.log_area.append("Starting inference...")
        self.progress.show()
        self.progress.setValue(0)
        self.info_label.setText("Status: Inference Running...")
        device = self.device_box.currentText()
        self.run_inference_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        self.load_csv_btn.setEnabled(False)
        self.save_pred_btn.setEnabled(False)
        compute_auc = self.auc_check.isChecked()
        multi_label = self.multi_label_check.isChecked()
        self.worker = InferenceWorker(self.model, loader, device, compute_auc=compute_auc, multi_label=multi_label)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log_message)
        self.worker.finished_signal.connect(self.inference_finished)
        self.worker.start()

    def update_progress(self, val):
        self.progress.setValue(val)

    def log_message(self, msg):
        self.log_area.append(msg)

    def inference_finished(self, status, acc, all_preds, all_labels, metrics_dict):
        self.run_inference_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.load_csv_btn.setEnabled(True)
        self.save_pred_btn.setEnabled(True)
        self.progress.hide()
        if status:
            self.preds = all_preds
            self.labels = all_labels
            self.extra_metrics = metrics_dict
            self.info_label.setText("Status: Inference Done")
            self.log_area.append(f"Inference Completed | Accuracy={acc:.4f}")
            if len(set(all_labels)) > 1:
                cm = confusion_matrix(all_labels, all_preds)
                cr = classification_report(all_labels, all_preds)
                self.log_area.append("Confusion Matrix:\n" + np.array2string(cm))
                self.log_area.append("Classification Report:\n" + cr)
            if "auc" in metrics_dict and metrics_dict["auc"] is not None:
                self.log_area.append(f"AUC={metrics_dict['auc']:.4f}")
        else:
            QMessageBox.critical(self, "Error", "Inference failed.")

    def save_predictions(self):
        if not self.preds:
            self.log_area.append("No predictions to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Predictions CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                arr = np.column_stack((self.preds, self.labels)) if self.labels else np.array(self.preds)
                header = "predictions,labels" if self.labels else "predictions"
                np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%d")
                self.log_area.append(f"Predictions saved to {os.path.basename(path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

def main():
    app = QApplication(sys.argv)
    w = InferenceWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
