# brain-tumor-detection

Brain tumor detection from MRI images using preprocessing, a rule-based image-processing baseline, classical machine learning, and a baseline CNN.

## What is included

- Dataset download helper
- Train/validation/test split preparation
- Image resizing and preprocessing
- Otsu thresholding baseline
- HOG feature extraction with SVM classification
- Baseline CNN training and evaluation
- Experiment log template with hardware and hyperparameter details
- Result artifacts under `results/`

## Requirements

- Python 3.12 or newer
- A terminal opened at the project root
- Internet access for the first dependency install and dataset download
- Optional: a CUDA or Apple Silicon MPS device for faster CNN training

- You can go step by step after installations or there is a full workflow under these steps, you may also use it.

## Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Optional: NVIDIA GPU Support (CUDA)

If you have an NVIDIA GPU and want to accelerate CNN training and evaluation with CUDA:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

## Install with uv

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Optional: NVIDIA GPU Support (CUDA)

If you have an NVIDIA GPU and want to accelerate CNN training and evaluation with CUDA:

```bash
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

## Prepare the data

The project expects the Kaggle dataset at:

```text
data/raw/brain-tumor-mri-data/versions/1/brain-tumor-mri-dataset
```

Download and place the dataset in the expected project path:

```bash
python src/data/download.py
```

The download script uses Kaggle's cache download and then copies the dataset into the `data/raw/.../brain-tumor-mri-dataset` path shown above.

Then create the split CSV files:

```bash
python -m src.data.prepare_pipeline
```

Then create the resized preprocessed images and CSV files:

```bash
python -m src.data.run_preprocessing 
```

After preprocessing, these files should exist:

```text
data/processed/train_preprocessed.csv
data/processed/val_preprocessed.csv
data/processed/test_preprocessed.csv
data/processed/images/
```

## Run every model

Run the Otsu image-processing baseline:

```bash
python -m src.baseline.otsu_baseline
```

Outputs are saved to:

```text
results/otsu_baseline/
```

Run the HOG + SVM classical machine-learning model:

```bash
python -m src.classical_ml.hog_svm
```

Outputs are saved to:

```text
results/classical_ml/hog_svm/
```

Train the baseline CNN:

```bash
python -m src.cnnModel.train
```

The best CNN weights are saved as:

```text
best_baseline_cnn.pth
```

Evaluate the trained CNN:

```bash
python -m src.cnnModel.evaluate
```

Run the CNN evaluation only after `best_baseline_cnn.pth` has been created by the training command.

## Typical full workflow

```bash
python src/data/download.py
python -m src.data.prepare_pipeline
python -m src.data.run_preprocessing
python -m src.baseline.otsu_baseline
python -m src.classical_ml.hog_svm
python -m src.cnnModel.train
python -m src.cnnModel.evaluate
```

## Notes

- The Otsu and HOG + SVM models use the preprocessed CSV files in `data/processed/`.
- The CNN dataloader reads the raw dataset from `data/raw/brain-tumor-mri-data/versions/1/brain-tumor-mri-dataset`.
- CNN training automatically uses CUDA, Apple Silicon MPS, or CPU depending on what PyTorch detects.
- Hardware details and model hyperparameters are summarized in `EXPERIMENT_LOGS.md`.
- If imports fail, make sure the virtual environment is active and dependencies were installed from `requirements.txt`.
