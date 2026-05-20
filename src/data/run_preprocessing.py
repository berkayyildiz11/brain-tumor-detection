from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATASET_DIR = Path(
    "data/raw/brain-tumor-mri-data/versions/1/brain-tumor-mri-dataset"
)

SPLIT_DIR = Path("data/processed")

TRAIN_CSV = SPLIT_DIR / "train_split.csv"
VAL_CSV = SPLIT_DIR / "val_split.csv"
TEST_CSV = SPLIT_DIR / "test_split.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 32


class BrainTumorCSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = Path(row["image_path"])
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def check_required_files():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_DIR}\n\n"
            "Please place the dataset in the correct data/raw location."
        )

    for csv_file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not csv_file.exists():
            raise FileNotFoundError(
                f"Split CSV not found:\n{csv_file}\n\n"
                "Please make sure committed split CSV files exist in data/processed/."
            )


def create_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    train_dataset = BrainTumorCSVDataset(TRAIN_CSV, transform=train_transform)
    val_dataset = BrainTumorCSVDataset(VAL_CSV, transform=eval_transform)
    test_dataset = BrainTumorCSVDataset(TEST_CSV, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def main():
    print("Running preprocessing pipeline...\n")

    check_required_files()

    train_loader, val_loader, test_loader = create_dataloaders()

    images, labels = next(iter(train_loader))

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("Preprocessing completed successfully.")
    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Device:", device)


if __name__ == "__main__":
    main()