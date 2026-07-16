# Rim Defect Detection — Krones Vision AI Challenge

Computer vision project for automotive wheel rim defect detection using both classical CV and deep learning approaches.

## Project Structure

```
CV/
├── main.py              # Real-time inspector (entry point)
├── evaluate.py          # Offline batch evaluation for inspector
├── core/                # Inspector core logic
│   └── bottle_inspector.py
├── utils/               # Inspector utilities (visualizer, smoother)
├── rim_classifier/      # Deep learning rim defect classifier
│   ├── preprocess.py    # Pre-crop rim images using Hough circles
│   ├── train_cnn.py     # Training v1: Single EfficientNet-B0
│   ├── train_cnn_v2.py  # Training v2: EnsembleModelV5 (main)
│   ├── train_kaggle.py  # Training v3: EnsembleModelV6 (Kaggle)
│   ├── predict.py       # Inference on test images
│   └── compare.py       # Post-training evaluation with metrics
├── assets/              # Trained model checkpoints
├── docs/                # Documentation & reports
├── notebooks/           # Jupyter notebooks
├── output/              # Generated output files (submission.csv)
├── runs/                # TensorBoard logs
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Preprocessing
```bash
python rim_classifier/preprocess.py --dataset path/to/train_images --cache-dir path/to/cropped_output
```

### Training
```bash
python rim_classifier/train_cnn_v2.py --dataset path/to/train_images --cache-dir path/to/cropped --batch 16 --img-size 256 --workers 4 --mixup-alpha 0.2
```

### Prediction
```bash
python rim_classifier/predict.py --test-dir path/to/test_data --threshold 0.5
```

### Evaluation
```bash
python rim_classifier/compare.py --model assets/model_v6.pt --threshold 0.4
```

### Real-time Inspector
```bash
python main.py --source 0
python evaluate.py --dataset path/to/test/folder
```

## Models

- **EnsembleModelV5**: EfficientNet-B2 + ResNet-50 + ConvNeXt-Tiny with CBAM attention
- **EnsembleModelV6**: EfficientNet-B4 + RegNetY-8GF + ConvNeXt-Small with cross-branch attention
