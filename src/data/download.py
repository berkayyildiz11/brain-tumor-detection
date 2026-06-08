from pathlib import Path
import kagglehub

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading dataset from Kaggle...")

path = kagglehub.dataset_download(
    "tombackert/brain-tumor-mri-data"
)

print(f"Dataset downloaded to: {path}")