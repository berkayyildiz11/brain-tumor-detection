from pathlib import Path
import csv

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.dataset import BrainTumorDataset

DATASET_DIR = Path(
    "data/raw/brain-tumor-mri-data/versions/1/brain-tumor-mri-dataset"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_STATE = 42


def save_split_csv(dataset, indices, filename):
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "class_name"])

        for idx in indices:
            image_path, label = dataset.samples[idx]
            class_name = dataset.class_names[label]
            writer.writerow([image_path, label, class_name])

    print(f"Saved: {output_path}")


def main():
    print("Preparing Brain Tumor MRI dataset pipeline...\n")

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

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    base_dataset = BrainTumorDataset(
        root_dir=DATASET_DIR,
        transform=None,
    )

    labels = [label for _, label in base_dataset.samples]

    print("Classes:", base_dataset.class_names)
    print("Class mapping:", base_dataset.class_to_idx)
    print("Total samples:", len(base_dataset))

    train_idx, test_idx = train_test_split(
        range(len(base_dataset)),
        test_size=0.15,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        stratify=[labels[i] for i in train_idx],
        random_state=RANDOM_STATE,
    )

    save_split_csv(base_dataset, train_idx, "train_split.csv")
    save_split_csv(base_dataset, val_idx, "val_split.csv")
    save_split_csv(base_dataset, test_idx, "test_split.csv")

    train_dataset = BrainTumorDataset(DATASET_DIR, transform=train_transform)
    eval_dataset = BrainTumorDataset(DATASET_DIR, transform=test_transform)

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        Subset(eval_dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        Subset(eval_dataset, test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("\nPipeline ready.")
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