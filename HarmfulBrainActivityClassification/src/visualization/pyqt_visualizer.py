import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTextEdit, QComboBox, QCheckBox, QMessageBox, QListWidget,
    QListWidgetItem, QDialog, QDialogButtonBox, QAbstractItemView
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt

class CSVLoaderThread(QThread):
    finished_signal = pyqtSignal(bool, str, list, dict)
    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                if not fieldnames:
                    self.finished_signal.emit(False, "CSV has no header or no data.", [], {})
                    return
                data_dict = {k: [] for k in fieldnames}
                for row in reader:
                    for k in fieldnames:
                        v = row.get(k, "")
                        try:
                            v = float(v)
                        except:
                            pass
                        data_dict[k].append(v)
            msg = f"Loaded {len(data_dict[fieldnames[0]])} rows from CSV."
            self.finished_signal.emit(True, msg, fieldnames, data_dict)
        except Exception as e:
            self.finished_signal.emit(False, str(e), [], {})

class MultiSelectionDialog(QDialog):
    def __init__(self, items, parent=None, title="Select Columns"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for i in items:
            self.list_widget.addItem(i)
        self.layout.addWidget(self.list_widget)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
    def get_selected(self):
        sel = []
        for it in self.list_widget.selectedItems():
            sel.append(it.text())
        return sel

class PyQtVisualizerWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Visualization (Advanced)")
        self.resize(1000, 700)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        top_layout = QHBoxLayout()
        self.load_csv_btn = QPushButton("Load CSV")
        self.plot_btn = QPushButton("Plot Curves")
        self.plot_btn.setEnabled(False)
        self.select_y_btn = QPushButton("Select Multiple Y")
        self.select_y_btn.setEnabled(False)
        top_layout.addWidget(self.load_csv_btn)
        top_layout.addWidget(self.select_y_btn)
        top_layout.addWidget(self.plot_btn)
        mid_layout = QHBoxLayout()
        self.xcol_label = QLabel("X-axis:")
        self.xcol_combo = QComboBox()
        self.ycol_label = QLabel("Y-axis:")
        self.ycol_combo = QComboBox()
        self.overlay_check = QCheckBox("Overlay Y2")
        self.y2col_label = QLabel("Y2-axis:")
        self.y2col_combo = QComboBox()
        self.y2col_combo.setEnabled(False)
        self.plot_type_label = QLabel("Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Line", "Scatter", "Bar"])
        mid_layout.addWidget(self.xcol_label)
        mid_layout.addWidget(self.xcol_combo)
        mid_layout.addWidget(self.ycol_label)
        mid_layout.addWidget(self.ycol_combo)
        mid_layout.addWidget(self.overlay_check)
        mid_layout.addWidget(self.y2col_label)
        mid_layout.addWidget(self.y2col_combo)
        mid_layout.addWidget(self.plot_type_label)
        mid_layout.addWidget(self.plot_type_combo)
        for w in [self.xcol_combo, self.ycol_combo, self.y2col_combo, self.overlay_check, self.plot_type_combo]:
            w.setEnabled(False)
        self.img_label = QLabel()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addLayout(top_layout)
        self.layout.addLayout(mid_layout)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.log_area)
        self.log_path = None
        self.fieldnames = []
        self.data_dict = {}
        self.y_columns = []
        self.load_csv_btn.clicked.connect(self.load_csv_file)
        self.plot_btn.clicked.connect(self.plot_curves)
        self.select_y_btn.clicked.connect(self.select_multiple_y)
        self.overlay_check.stateChanged.connect(self.toggle_overlay)

    def toggle_overlay(self, state):
        self.y2col_combo.setEnabled(state == 2)

    def load_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if not path:
            return
        self.log_path = path
        self.log_area.append(f"Loading CSV: {path}")
        self.loader_thread = CSVLoaderThread(path)
        self.loader_thread.finished_signal.connect(self.on_csv_loaded)
        self.loader_thread.start()

    def on_csv_loaded(self, success, message, fieldnames, data_dict):
        if not success:
            QMessageBox.critical(self, "Error", message)
            self.log_area.append(f"Error: {message}")
            return
        self.fieldnames = fieldnames
        self.data_dict = data_dict
        self.log_area.append(message)
        self.xcol_combo.clear()
        self.ycol_combo.clear()
        self.y2col_combo.clear()
        for fn in fieldnames:
            self.xcol_combo.addItem(fn)
            self.ycol_combo.addItem(fn)
            self.y2col_combo.addItem(fn)
        if len(fieldnames) > 0:
            self.xcol_combo.setCurrentIndex(0)
        if len(fieldnames) > 1:
            self.ycol_combo.setCurrentIndex(1)
        for w in [self.xcol_combo, self.ycol_combo, self.overlay_check, self.plot_type_combo]:
            w.setEnabled(True)
        self.plot_btn.setEnabled(True)
        self.select_y_btn.setEnabled(True)

    def select_multiple_y(self):
        if not self.fieldnames:
            self.log_area.append("No CSV data loaded.")
            return
        dialog = MultiSelectionDialog(self.fieldnames, self, title="Select Y Columns")
        if dialog.exec_() == QDialog.Accepted:
            self.y_columns = dialog.get_selected()
            self.log_area.append(f"Selected multiple Y columns: {self.y_columns}")

    def to_float_array(self, arr):
        try:
            return np.array(arr, dtype=float)
        except:
            return None

    def plot_curves(self):
        if not self.fieldnames or not self.data_dict:
            self.log_area.append("No data to plot.")
            return
        xcol = self.xcol_combo.currentText()
        if xcol not in self.data_dict:
            self.log_area.append("Invalid X column.")
            return
        xarr = self.to_float_array(self.data_dict[xcol])
        if xarr is None:
            self.log_area.append(f"Could not convert X column {xcol} to float.")
            return
        overlay = self.overlay_check.isChecked()
        plot_type = self.plot_type_combo.currentText().lower()
        if overlay:
            y2col = self.y2col_combo.currentText()
            if y2col not in self.data_dict:
                self.log_area.append("Y2-axis column not valid.")
                return
            y2arr = self.to_float_array(self.data_dict[y2col])
            if y2arr is None:
                self.log_area.append(f"Could not convert Y2 column {y2col} to float.")
                return
        else:
            y2arr = None
        ycol = self.ycol_combo.currentText()
        if ycol not in self.data_dict:
            self.log_area.append("Invalid Y column.")
            return
        yarr = self.to_float_array(self.data_dict[ycol])
        if yarr is None:
            self.log_area.append(f"Could not convert Y column {ycol} to float.")
            return
        plt.figure(figsize=(8, 5))
        if len(self.y_columns) > 1:
            for yc in self.y_columns:
                if yc not in self.data_dict:
                    self.log_area.append(f"Column {yc} invalid.")
                    continue
                yvals = self.to_float_array(self.data_dict[yc])
                if yvals is None:
                    self.log_area.append(f"Could not convert {yc} to float.")
                    continue
                if plot_type == "line":
                    plt.plot(xarr, yvals, label=yc)
                elif plot_type == "scatter":
                    plt.scatter(xarr, yvals, label=yc)
                else:
                    width = (max(xarr) - min(xarr)) / (len(xarr) * 2)
                    plt.bar(xarr, yvals, width=width, label=yc)
            plt.xlabel(xcol)
            if not overlay:
                plt.ylabel("Values")
            plt.title("Multi Y Plot")
            plt.legend()
        else:
            if plot_type == "line":
                plt.plot(xarr, yarr, label=ycol, color="blue")
            elif plot_type == "scatter":
                plt.scatter(xarr, yarr, label=ycol, color="blue")
            else:
                width = (max(xarr) - min(xarr)) / (len(xarr) * 2) if len(xarr) > 1 else 0.1
                plt.bar(xarr, yarr, width=width, label=ycol, color="blue")
            if overlay and y2arr is not None:
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                if plot_type == "line":
                    ax2.plot(xarr, y2arr, label=y2col, color="red")
                elif plot_type == "scatter":
                    ax2.scatter(xarr, y2arr, label=y2col, color="red")
                else:
                    width2 = (max(xarr) - min(xarr)) / (len(xarr) * 2)
                    ax2.bar(xarr+width2, y2arr, width=width2, label=y2col, color="red")
                ax1.set_xlabel(xcol)
                ax1.set_ylabel(ycol, color="blue")
                ax2.set_ylabel(y2col, color="red")
            else:
                plt.xlabel(xcol)
                plt.ylabel(ycol)
        plt.grid(True)
        plt.tight_layout()
        output_path = "training_plot.png"
        plt.savefig(output_path)
        plt.close()
        self.img_label.setPixmap(QPixmap(output_path))
        if overlay and y2arr is not None:
            self.log_area.append(f"Plotted columns: X={xcol}, Y={ycol}, Y2={y2col}, PlotType={plot_type}")
        elif len(self.y_columns) > 1:
            self.log_area.append(f"Plotted multiple Y: {self.y_columns}, X={xcol}, PlotType={plot_type}")
        else:
            self.log_area.append(f"Plotted columns: X={xcol}, Y={ycol}, PlotType={plot_type}")

def main():
    app = QApplication(sys.argv)
    window = PyQtVisualizerWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
