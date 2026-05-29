from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATASET_DIR = Path(
    r"C:\Users\Hüseyin Yorga\Documents\GitHub\brain-tumor-detection\brain-tumor-mri-dataset"
)

SPLIT_DIR = Path("data/processed")

TRAIN_CSV = SPLIT_DIR / "train_split.csv"
VAL_CSV = SPLIT_DIR / "val_split.csv"
TEST_CSV = SPLIT_DIR / "test_split.csv"

PREPROCESSED_IMAGE_DIR = SPLIT_DIR / "images"

TRAIN_PREPROCESSED_CSV = SPLIT_DIR / "train_preprocessed.csv"
VAL_PREPROCESSED_CSV = SPLIT_DIR / "val_preprocessed.csv"
TEST_PREPROCESSED_CSV = SPLIT_DIR / "test_preprocessed.csv"

IMAGE_SIZE = 224
BATCH_SIZE = 32


LABEL_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary",
}


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


def preprocess_and_save_split(input_csv, split_name, output_csv):
    df = pd.read_csv(input_csv)

    output_rows = []
    split_image_dir = PREPROCESSED_IMAGE_DIR / split_name
    split_image_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        original_path = Path(row["image_path"])
        label = int(row["label"])
        label_name = LABEL_NAMES[label]

        class_dir = split_image_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(original_path).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

        new_image_name = f"{split_name}_{idx}_{original_path.stem}.png"
        new_image_path = class_dir / new_image_name

        image.save(new_image_path)

        output_rows.append({
            "image_path": str(new_image_path),
            "label": label,
            "label_name": label_name,
            "original_image_path": str(original_path),
        })

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv, index=False)

    print(f"{split_name} images saved to: {split_image_dir}")
    print(f"{split_name} CSV saved to: {output_csv}")


def save_preprocessed_images():
    preprocess_and_save_split(
        input_csv=TRAIN_CSV,
        split_name="train",
        output_csv=TRAIN_PREPROCESSED_CSV,
    )

    preprocess_and_save_split(
        input_csv=VAL_CSV,
        split_name="val",
        output_csv=VAL_PREPROCESSED_CSV,
    )

    preprocess_and_save_split(
        input_csv=TEST_CSV,
        split_name="test",
        output_csv=TEST_PREPROCESSED_CSV,
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

    train_dataset = BrainTumorCSVDataset(
        TRAIN_PREPROCESSED_CSV,
        transform=train_transform,
    )

    val_dataset = BrainTumorCSVDataset(
        VAL_PREPROCESSED_CSV,
        transform=eval_transform,
    )

    test_dataset = BrainTumorCSVDataset(
        TEST_PREPROCESSED_CSV,
        transform=eval_transform,
    )

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

    save_preprocessed_images()

    train_loader, val_loader, test_loader = create_dataloaders()

    images, labels = next(iter(train_loader))

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("\nPreprocessing completed successfully.")
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