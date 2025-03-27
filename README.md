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
