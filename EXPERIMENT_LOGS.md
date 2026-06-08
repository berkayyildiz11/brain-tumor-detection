# Experiment Logs

This file summarizes the planned experiment environment and the hyperparameters used by each model in the project.

## Hardware and Environment

Fill this section after running the experiments on the final machine.

| Item | Value |
| --- | --- |
| Operating system | Windows / macOS / Linux |
| Python version | 3.12.x |
| CPU | To be recorded |
| GPU | To be recorded, or CPU-only |
| RAM | To be recorded |
| PyTorch device used | `cuda`, `mps`, or `cpu` |
| Dataset | Brain Tumor MRI dataset from Kaggle |

The CNN training and evaluation scripts automatically select the best available device:

1. NVIDIA CUDA GPU, if available
2. Apple Silicon MPS, if available
3. CPU fallback

## Data Preparation Settings

| Setting | Value |
| --- | --- |
| Raw dataset path | `data/raw/brain-tumor-mri-data/versions/1/brain-tumor-mri-dataset` |
| Processed data path | `data/processed/` |
| Image size | `224 x 224` |
| Batch size | `32` |
| Random state | `42` |
| Test split | `15%` |
| Validation split | `15%` of the remaining training data |
| Classes | `glioma`, `meningioma`, `notumor`, `pituitary` |

## Otsu Baseline

| Hyperparameter / Setting | Value |
| --- | --- |
| Input split | `data/processed/test_preprocessed.csv` |
| Labeling task | Binary tumor vs. no tumor |
| Image mode | Grayscale |
| Gaussian blur kernel | `(5, 5)` |
| Thresholding method | Otsu thresholding |
| Edge detector | Canny |
| Canny thresholds | `100`, `200` |
| Rule: white pixel ratio | `> 0.25` |
| Rule: contour area | `> 3000` |
| Rule: edge density | `> 0.035` |
| Output directory | `results/otsu_baseline/` |

## HOG + SVM Model

| Hyperparameter / Setting | Value |
| --- | --- |
| Input image size | `128 x 128` |
| HOG orientations | `9` |
| HOG pixels per cell | `(16, 16)` |
| HOG cells per block | `(2, 2)` |
| HOG block normalization | `L2-Hys` |
| Feature scaling | `StandardScaler` |
| Classifier | Support Vector Machine |
| SVM kernel | `rbf` |
| SVM C | `10` |
| SVM gamma | `scale` |
| Grid search | Disabled by default |
| Grid search CV | `3`, if enabled |
| Output directory | `results/classical_ml/hog_svm/` |

## Baseline CNN

| Hyperparameter / Setting | Value |
| --- | --- |
| Input image size | `224 x 224` |
| Batch size | `32` |
| Epochs | `10` |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Loss function | Cross entropy loss |
| Data augmentation | Random horizontal flip, random rotation |
| Random rotation | `10` degrees |
| Normalization mean | `[0.5, 0.5, 0.5]` |
| Normalization std | `[0.5, 0.5, 0.5]` |
| Dropout | `0.5` |
| Saved model | `best_baseline_cnn.pth` |

### CNN Architecture

| Layer | Configuration |
| --- | --- |
| Convolution 1 | `3 -> 32`, kernel `3`, padding `1` |
| Pooling | MaxPool, kernel `2`, stride `2` |
| Convolution 2 | `32 -> 64`, kernel `3`, padding `1` |
| Pooling | MaxPool, kernel `2`, stride `2` |
| Convolution 3 | `64 -> 128`, kernel `3`, padding `1` |
| Pooling | MaxPool, kernel `2`, stride `2` |
| Fully connected 1 | `128 * 28 * 28 -> 512` |
| Dropout | `0.5` |
| Fully connected 2 | `512 -> number_of_classes` |

## Metrics to Report

| Model | Metrics / Artifacts |
| --- | --- |
| Otsu baseline | Accuracy, macro F1-score, confusion matrix, classification report |
| HOG + SVM | Accuracy, macro AUC, confusion matrices, per-class precision/recall/F1 |
| Baseline CNN | Training loss, training accuracy, validation loss, validation accuracy, test classification report |

## Run Commands

```bash
python src/data/download.py
python src/data/prepare_pipeline.py
python src/data/run_preprocessing.py
python src/baseline/otsu_baseline.py
python src/classical_ml/hog_svm.py
python src/cnnModel/train.py
python src/cnnModel/evaluate.py
```
