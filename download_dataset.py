import shutil
from pathlib import Path

import kagglehub

def main():
    print("Downloading dataset from Kaggle...")
    # Downloads the dataset to a local cache directory
    cache_path = kagglehub.dataset_download("tombackert/brain-tumor-mri-data")
    print(f"Downloaded to cache: {cache_path}")
    
    # The exact path where your friend's dataloaders expect the data to be
    target_dir = Path(r"C:\Users\Hüseyin Yorga\Documents\GitHub\brain-tumor-detection\brain-tumor-mri-dataset")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying and merging files to {target_dir}...")
    # The Kaggle dataset separates images into "Training" and "Testing" folders.
    # This merges them so the dataloader can properly create its own splits.
    for split in ["Training", "Testing"]:
        split_dir = Path(cache_path) / split
        if split_dir.exists():
            shutil.copytree(split_dir, target_dir, dirs_exist_ok=True)
            
    print("Done! You can now run your training script.")

if __name__ == "__main__":
    main()