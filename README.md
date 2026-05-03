# bottle_classifier
his project is an automated industrial quality control system designed to detect defects on bottle rims (sealing surfaces) using Computer Vision and Deep Learning. It utilizes a high-speed preprocessing pipeline followed by an ensemble CNN architecture to identify chips, cracks, and other anomalies.  
##📂 Project StructurePlaintext├── assets/             
# Model weights (model.pt) and validation metadata
├── core/               # Core inspection logic (bottle_inspector.py)[cite: 4, 5]
├── utils/              # UI and smoothing helpers
├── runs/               # TensorBoard training logs[cite: 9, 10]
├── preprocess.py       # High-speed parallel rim cropping[cite: 7]
├── train_cnn.py        # Standard EfficientNet-B0 training
├── train_cnn_v2.py     # High-end Ensemble (B2 + ConvNeXt) training
├── main.py             # Real-time camera inspection entry point
├── predict.py          # Inference script for test image sets[cite: 6]
└── compare.py          # Validation metrics and threshold optimization[cite: 2]
## 🚀 How to Use1. PreprocessingBefore training, run the preprocessing script to detect and crop bottle rims. This significantly speeds up training by caching localized images.  Bashpython preprocess.py --dataset path/to/images --cache-dir path/to/output
2. TrainingChoose your training pipeline based on your hardware:
  Standard Training: Use train_cnn.py for a lightweight EfficientNet-B0 model.
  High-End Training: If you have a high-end graphics card, it is recommended to use train_cnn_v2.py. This script utilizes an ensemble of EfficientNet-B2 and ConvNeXt-Tiny, which typically yields 2% higher accuracy than the regular version
3. Real-Time InspectionTo run the inspector on a live camera feed or video file:
    Bashpython main.py --source 0 --chip-threshold 200
Press 'C' to calibrate a baseline "Good" frame.
Press 'Q' to quit.
4. EvaluationUse compare.py to evaluate the model on your validation set and perform a threshold sweep to find the optimal F1-score for your specific production line.
## 📊 Performance Tracking
Training metrics(Loss, Accuracy, F1) are logged for TensorBoard. To view your logs in the runs/ folder, run:  Bashtensorboard --logdir=runs
## [!IMPORTANT]Project Status: 
This project is still actively in development. New updates regarding model optimization and real-time processing speed are added regularly.
