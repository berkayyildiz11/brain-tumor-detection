from torchvision import transforms

from src.data.dataset import BrainTumorDataset

DATASET_DIR = "data/raw/brain-tumor-mri-data/versions/1/brain-tumor-mri-dataset"


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = BrainTumorDataset(
        root_dir=DATASET_DIR,
        transform=transform,
    )

    image, label = dataset[0]

    print("Classes:", dataset.class_names)
    print("Class mapping:", dataset.class_to_idx)
    print("Total samples:", len(dataset))
    print("Image tensor shape:", image.shape)
    print("Label:", label)


if __name__ == "__main__":
    main()