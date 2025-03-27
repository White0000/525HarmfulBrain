import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from src.data_loader.dataset_manager import DatasetManagerWindow, DataManager
from src.trainer.train_pipeline import TrainPipelineWindow
from src.inference.inference_pipeline import InferenceWindow
from src.visualization.pyqt_visualizer import PyQtVisualizerWindow
from src.visualization.open3d_visualizer import Open3DVisualizerWindow
from src.data_preprocessing.preprocess_pipeline import PreprocessWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shared_manager = DataManager()
        self.tabs = QTabWidget()
        self.dataset_tab = DatasetManagerWindow(manager=self.shared_manager)
        self.preprocess_tab = PreprocessWindow()
        self.train_tab = TrainPipelineWindow(manager=self.shared_manager)
        self.inference_tab = InferenceWindow(manager=self.shared_manager)
        self.visual_tab = PyQtVisualizerWindow()
        self.open3d_tab = Open3DVisualizerWindow()
        self.tabs.addTab(self.dataset_tab, "Dataset")
        self.tabs.addTab(self.preprocess_tab, "Preprocessing")
        self.tabs.addTab(self.train_tab, "Training")
        self.tabs.addTab(self.inference_tab, "Inference")
        self.tabs.addTab(self.visual_tab, "2D Viz")
        self.tabs.addTab(self.open3d_tab, "3D Viz")
        self.setCentralWidget(self.tabs)
        self.setWindowTitle("Harmful Brain Activity Classification")
        self.resize(1000, 600)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
