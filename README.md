# bottle_classifier

> Automated industrial quality control for bottle rim defect detection — chips, cracks, and sealing surface anomalies — using a high-speed preprocessing pipeline and ensemble CNN architecture.

![Status](https://img.shields.io/badge/status-active_development-brightgreen)
![Model](https://img.shields.io/badge/model-EfficientNet_B2_+_ConvNeXt--Tiny-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)

---

## Overview

This project implements a real-time computer vision pipeline for detecting defects on bottle rims (sealing surfaces) in industrial settings. It combines a high-speed parallel preprocessing stage with an ensemble CNN architecture to reliably identify chips, cracks, and other surface anomalies.

**Two training pipelines are available:**

| Pipeline | Model | Hardware | Notes |
|---|---|---|---|
| Standard | EfficientNet-B0 | Any GPU | Lightweight, fast training |
| Ensemble (v2) | EfficientNet-B2 + ConvNeXt-Tiny | High-end GPU | ~2% higher accuracy — recommended |

---

## Project Structure

```
bottle_classifier/
├── assets/             # Model weights (model.pt) and validation metadata
├── core/               # Core inspection logic (bottle_inspector.py)
├── utils/              # UI and smoothing helpers
├── runs/               # TensorBoard training logs
├── preprocess.py       # High-speed parallel rim cropping
├── train_cnn.py        # Standard EfficientNet-B0 training
├── train_cnn_v2.py     # Ensemble (B2 + ConvNeXt-Tiny) training
├── main.py             # Real-time camera inspection entry point
├── predict.py          # Batch inference script for test image sets
└── compare.py          # Validation metrics and threshold optimization
```

---

## Quick Start

### 1. Preprocess Dataset

Detect and crop bottle rims from your raw images. This caches localized crops and significantly speeds up training.

```bash
python preprocess.py --dataset path/to/images --cache-dir path/to/output
```

### 2. Train the Model

Choose your pipeline based on available hardware:

```bash
# Standard — lightweight, works on any GPU
python train_cnn.py

# Ensemble — recommended for high-end GPUs (~2% accuracy gain)
python train_cnn_v2.py
```

### 3. Real-Time Inspection

Run the inspector on a live camera feed or video file:

```bash
python main.py --source 0 --chip-threshold 200
```

| Key | Action |
|---|---|
| `C` | Calibrate baseline from a known-good frame |
| `Q` | Quit |

### 4. Evaluate & Optimize Threshold

Run validation metrics and sweep thresholds to find the optimal F1-score cutoff for your specific production line:

```bash
python compare.py
```

---

## Performance Tracking

Training metrics (loss, accuracy, F1) are logged for TensorBoard. To view your training runs:

```bash
tensorboard --logdir=runs
```

---

## Batch Inference

To run inference on a set of test images without the live camera feed:

```bash
python predict.py --source path/to/test/images
```

---

> [!IMPORTANT]
> **Project Status:** Actively in development. Updates to model optimization and real-time processing speed are added regularly. Expect API changes between versions.
