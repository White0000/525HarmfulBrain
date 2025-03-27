import sys
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QTextEdit, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSlider, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

class CSVLoaderThread(QThread):
    finished_signal = pyqtSignal(bool, str, object)
    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        try:
            arr = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    vals = line.strip().split(",")
                    if len(vals) >= 3:
                        try:
                            x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                            arr.append([x, y, z])
                        except:
                            pass
            if not arr:
                self.finished_signal.emit(False, "No valid 3D points found.", None)
            else:
                pts = np.array(arr, dtype=np.float32)
                msg = f"Loaded {len(pts)} points from {self.path}"
                self.finished_signal.emit(True, msg, pts)
        except Exception as e:
            self.finished_signal.emit(False, str(e), None)

class Open3DVisualizerWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open3D Visualizer (Advanced)")
        self.resize(1000, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        top_layout = QHBoxLayout()
        self.load_csv_btn = QPushButton("Load 3D Points CSV")
        self.downsample_check = QCheckBox("Voxel Downsample")
        self.downsample_check.setToolTip("Reduce point density for large point clouds.")
        self.downsample_check.setChecked(False)
        self.outlier_check = QCheckBox("Remove Outliers")
        self.outlier_check.setToolTip("Statistical outlier removal.")
        self.outlier_check.setChecked(False)
        self.normals_check = QCheckBox("Compute Normals")
        self.normals_check.setToolTip("Estimate normals for the point cloud.")
        self.normals_check.setChecked(False)
        self.show_axis_check = QCheckBox("Show Axis Frame")
        self.show_axis_check.setToolTip("Display an axis coordinate frame at origin.")
        self.show_axis_check.setChecked(False)
        self.color_mode_label = QLabel("Color:")
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["Uniform (Cyan)", "Random Colors", "By Z Value"])
        self.point_size_label = QLabel("Point Size:")
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setMinimum(1)
        self.point_size_slider.setMaximum(10)
        self.point_size_slider.setValue(3)
        top_layout.addWidget(self.load_csv_btn)
        top_layout.addWidget(self.downsample_check)
        top_layout.addWidget(self.outlier_check)
        top_layout.addWidget(self.normals_check)
        top_layout.addWidget(self.show_axis_check)
        top_layout.addWidget(self.color_mode_label)
        top_layout.addWidget(self.color_mode_combo)
        top_layout.addWidget(self.point_size_label)
        top_layout.addWidget(self.point_size_slider)
        self.layout.addLayout(top_layout)
        self.para_layout = QHBoxLayout()
        self.outlier_nb_label = QLabel("NB Neighbors:")
        self.outlier_nb_edit = QLineEdit("20")
        self.outlier_std_label = QLabel("Std Ratio:")
        self.outlier_std_edit = QLineEdit("2.0")
        self.voxel_label = QLabel("Voxel Size:")
        self.voxel_edit = QLineEdit("0.05")
        self.normals_rad_label = QLabel("Normal Radius:")
        self.normals_rad_edit = QLineEdit("0.1")
        self.para_layout.addWidget(self.outlier_nb_label)
        self.para_layout.addWidget(self.outlier_nb_edit)
        self.para_layout.addWidget(self.outlier_std_label)
        self.para_layout.addWidget(self.outlier_std_edit)
        self.para_layout.addWidget(self.voxel_label)
        self.para_layout.addWidget(self.voxel_edit)
        self.para_layout.addWidget(self.normals_rad_label)
        self.para_layout.addWidget(self.normals_rad_edit)
        self.layout.addLayout(self.para_layout)
        self.vis_btn = QPushButton("Visualize 3D")
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(self.vis_btn)
        self.layout.addWidget(self.log_area)
        self.points = None
        self.csv_thread = None
        self.load_csv_btn.clicked.connect(self.load_csv)
        self.vis_btn.clicked.connect(self.show_3d)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if not path:
            return
        self.log_area.append("Loading CSV in background...")
        self.csv_thread = CSVLoaderThread(path)
        self.csv_thread.finished_signal.connect(self.on_csv_loaded)
        self.csv_thread.start()

    def on_csv_loaded(self, success, message, data):
        if success:
            self.points = data
            self.log_area.append(message)
        else:
            self.log_area.append(f"Error: {message}")

    def show_3d(self):
        if self.points is None or len(self.points) == 0:
            self.log_area.append("No points loaded.")
            return
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points.copy())
        if self.downsample_check.isChecked():
            try:
                voxel_size = float(self.voxel_edit.text())
            except:
                voxel_size = 0.05
            pc = pc.voxel_down_sample(voxel_size)
            self.log_area.append(f"Downsampled with voxel_size={voxel_size}. New size = {len(pc.points)}")
        if self.outlier_check.isChecked():
            try:
                nb_neighbors = int(self.outlier_nb_edit.text())
            except:
                nb_neighbors = 20
            try:
                std_ratio = float(self.outlier_std_edit.text())
            except:
                std_ratio = 2.0
            ind = pc.remove_statistical_outlier(nb_neighbors, std_ratio)[1]
            pc = pc.select_by_index(ind)
            self.log_area.append(f"Removed outliers (nb={nb_neighbors}, std={std_ratio}). New size={len(pc.points)}")
        if self.normals_check.isChecked():
            try:
                rad = float(self.normals_rad_edit.text())
            except:
                rad = 0.1
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=30))
            self.log_area.append(f"Computed normals with radius={rad}.")
        color_mode = self.color_mode_combo.currentText()
        if color_mode.startswith("Uniform"):
            pc.paint_uniform_color([0.2, 0.6, 0.8])
        elif color_mode.startswith("Random"):
            c = np.random.rand(len(pc.points), 3)
            pc.colors = o3d.utility.Vector3dVector(c)
        else:
            coords = np.asarray(pc.points)
            z_vals = coords[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            rng = z_max - z_min
            if rng < 1e-8:
                rng = 1e-8
            norm_z = (z_vals - z_min) / rng
            colors = np.zeros((len(pc.points), 3), dtype=np.float32)
            colors[:, 0] = 1.0 - norm_z
            colors[:, 1] = norm_z
            pc.colors = o3d.utility.Vector3dVector(colors)
        geometries = [pc]
        if self.show_axis_check.isChecked():
            axis_len = 0.5
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_len, origin=[0, 0, 0])
            geometries.append(axis)
        ps = self.point_size_slider.value()
        self.log_area.append(f"Launching Open3D window with point_size={ps}...")
        self.launch_custom_visualizer(geometries, ps)

    def launch_custom_visualizer(self, geometries, point_size=3):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Open3D Visualization", width=1280, height=720)
        for g in geometries:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.asarray([0, 0, 0], dtype=np.float32)
        vc = vis.get_view_control()
        vc.set_zoom(0.8)
        vis.run()
        vis.destroy_window()

def main():
    app = QApplication(sys.argv)
    w = Open3DVisualizerWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
