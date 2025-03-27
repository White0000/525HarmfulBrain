# Harmful Brain Activity Classification

This repository provides a complete deep learning pipeline for classifying harmful brain activities from EEG signals, such as Seizure, GPD, LRDA, GRDA, or other abnormal patterns. It offers an end-to-end solution, including data loading, preprocessing, model training (CNN/LSTM/Transformer), inference, and 2D/3D visualization.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Installation & Environment](#installation--environment)
4. [Data Organization](#data-organization)
5. [Usage](#usage)
   1. [1. Dataset Management](#1-dataset-management)
   2. [2. Preprocessing (Optional)](#2-preprocessing-optional)
   3. [3. Training](#3-training)
   4. [4. Inference](#4-inference)
   5. [5. Visualization (2D & 3D)](#5-visualization-2d--3d)
6. [Model Architecture](#model-architecture)
7. [Results & Performance](#results--performance)
8. [Potential Extensions](#potential-extensions)
9. [License](#license)
10. [Contact](#contact)

---

## Project Overview

The **Harmful Brain Activity Classification** project aims to identify various EEG abnormalities in a hospital or research context (e.g., ICU). Common target classes include:
- **Seizure**
- **GPD/LPD** (Generalized or Lateralized Periodic Discharges)
- **LRDA/GRDA** (Lateralized or Generalized Rhythmic Delta Activity)
- **Other**  

We unify these into numerical labels, train neural networks (CNN, LSTM, Transformer), and achieve high accuracy and near-perfect AUC on large EEG datasets.

---

## Key Features

1. **Flexible DataLoader**: Easily load EEG CSV with multiple numeric columns (offsets, votes, etc.) and label columns (expert_consensus or derived from votes).
2. **Optional Preprocessing**: GUI-based filtering (bandpass, wavelet, notch), segmentation, saving processed CSV.
3. **Modular Models**: Choose from:
   - **Linear** (simple baseline)
   - **CNN** (2D residual + attention blocks + mini-Transformer)
   - **LSTM** (with multi-head self-attention)
   - **Transformer** (positional encoding, multi-layer encoder)
4. **Training Pipeline**:
   - PyTorch-based training loop
   - Live logging of each epoch’s loss & accuracy
   - Automatic saving of the full model (for easy `torch.load(...)`)
5. **Inference Pipeline**:
   - Load model `.pth`, load inference CSV
   - Run predictions, print classification report, confusion matrix, AUC
   - Optionally save predictions
6. **Visualization**:
   - **2D**: Plot CSV columns (line/scatter/bar) with overlay Y2
   - **3D**: View 3D point cloud in Open3D (downsample, remove outliers, color, etc.)

---

## Installation & Environment

1. **Clone or Download** this repository.
2. **Install Python 3.8+** (recommended).
3. Create a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - Includes PyTorch, PyQt5, NumPy, SciPy, scikit-learn, open3d, etc.
5. (Optional) For GPU training, ensure **CUDA** is available and install a CUDA-enabled PyTorch.

---

## Data Organization

Inside `data/` directory (or your chosen path), we suggest the following structure:

```
data/
├─ raw/
│   └─ train.csv  (original EEG data)
├─ processed/
│   └─ train_processed.csv (optionally pre-cleaned)
└─ README.md (notes about data)
```

Each CSV row typically contains:
- **feature columns**: time offsets, multiple vote columns
- **label** column: if already assigned (or else generate via advanced labeling)

---

## Usage

You can run the GUI with:
```bash
python main.py
```
This will open a PyQt-based interface with the following tabs:

### 1. Dataset Management

- **Load CSV**: Select your `.csv`. The system prints how many rows loaded, which columns exist, etc.
- **Advanced Labeling**: Merge multiple columns (like `seizure_vote`, `lpd_vote`, `gpd_vote`...) into a single label or use `expert_consensus`.
- **Select Features**: Pick which columns become features.
- **Create DataLoader**: Optionally apply bandpass filters or random noise. Finally, see the shape of first batch, distribution of classes.

### 2. Preprocessing (Optional)

If your data is raw EEG signals:
- **Apply Filter**: bandpass (1~30Hz), wavelet denoise, etc.
- **Segment Data**: choose a segment size in sample points.
- **Save CSV**: store the processed results for training.

(*This step is not mandatory if your data is already cleaned.*)

### 3. Training

- **Load Dataset from Manager**: Fetch the dataset loaded in the previous tab.
- **Device**: CPU or CUDA (if available).
- **Model**: choose among Linear, CNN, LSTM, Transformer.
- **Hyperparams**: LR, Epochs, Optimizer, WeightDecay, Momentum, Scheduler.
- **Start Training**: watch live logs of epoch/loss/accuracy. After finishing, it prompts whether to save the entire model (`.pth`).

### 4. Inference

- **Load Trained Model**: pick your saved `.pth` file.
- **Load Inference CSV**: dataset to infer on.
- **Run Inference**: prints accuracy, confusion matrix, classification report, optional AUC.  
- **Save Predictions**: store predicted labels in a CSV.

### 5. Visualization (2D & 3D)

**2D Viz**:
- **Load CSV**: any CSV with numeric columns.
- **X-axis, Y-axis**: pick columns to visualize (e.g., `eeg_id` vs. `gpd_vote`).
- **Overlay Y2**: a second axis if needed.
- **Plot Type**: line, scatter, or bar.
- Renders a static `.png`, displayed within the GUI.

**3D Viz**:
- **Load 3D Points CSV**: must have `[x, y, z]` columns.
- **Voxel Downsample**, **Remove Outliers**, etc.
- **Visualize 3D**: opens an Open3D window for interactive point cloud exploration.

---

## Model Architecture

We provide four main model families:

1. **Linear**: A simple fully connected network, for quick baselines.
2. **CNN**: A 2D convolution approach with residual blocks, channel/spatial attention, plus a mini-Transformer block in the final layers.
3. **LSTM**: A sequence-based classifier with multi-head self-attention. Accepts `(batch, seq_len, input_dim)`.
4. **Transformer**: Standard encoder block with positional encoding, multi-head attention, and feed-forward sublayers.

Each can handle either a “(batch, F)” or “(batch, seq_len, input_dim)” shape. They reshape or interpret the data accordingly.

---

## Results & Performance

- **Accuracy**: Often 97%~99%+ on the sample EEG dataset (over 100k samples).  
- **AUC**: Commonly ≥ 0.99.  
- **Runtime**: Achieves quick convergence (within 10–30 epochs) if the dataset columns strongly correlate with the final label.  
- See the logs in “Training” or “Inference” tabs for epoch-by-epoch improvement and final metrics.

---

## Potential Extensions

1. **External Validation**: Evaluate on a separate unseen test set to confirm no overfitting.  
2. **More Preprocessing**: Add advanced wavelet denoising or artifact removal.  
3. **Multi-label**: If some EEG segments can have multiple co-occurring patterns, adapt the dataset/labels for a multi-label approach.  
4. **Real-time**: Convert the final model to TorchScript or ONNX for streaming inference.

---

## License

MIT License

---

## Contact

For questions or further support, please contact:

- **Name**: [Baihe]  
- **Email**: []  
- **Institution**: [LU]

Thank you for using **Harmful Brain Activity Classification**. We hope it helps improve EEG analysis and patient outcomes.
