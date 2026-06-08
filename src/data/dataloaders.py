from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.data.dataset import BrainTumorDataset

DATASET_DIR = Path(
    r"C:\Users\Hüseyin Yorga\Documents\GitHub\brain-tumor-detection\brain-tumor-mri-dataset"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_STATE = 42


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


def create_dataloaders():
    full_dataset = BrainTumorDataset(
        root_dir=DATASET_DIR,
        transform=None,
    )

    labels = [label for _, label in full_dataset.samples]

    # 1. First Split: Isolate 15% of the data for the Test Set.
    # The remaining 85% goes into train_idx.
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.15,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    # 2. Second Split: Take the remaining 85% and split it again.
    # 15% of this remaining batch becomes the Validation Set.
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        stratify=[labels[i] for i in train_idx],
        random_state=RANDOM_STATE,
    )

    train_dataset = BrainTumorDataset(
        root_dir=DATASET_DIR,
        transform=train_transform,
    )

    test_dataset = BrainTumorDataset(
        root_dir=DATASET_DIR,
        transform=test_transform,
    )

    train_dataset = Subset(train_dataset, train_idx)
    val_dataset = Subset(test_dataset, val_idx)
    test_dataset = Subset(test_dataset, test_idx)

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

    return (
        train_loader,
        val_loader,
        test_loader,
        full_dataset.class_names,
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = create_dataloaders()

    print("Classes:", class_names)

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)
