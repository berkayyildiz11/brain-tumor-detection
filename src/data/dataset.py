from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.class_names = sorted(
            [p.name for p in self.root_dir.iterdir() if p.is_dir()]
        )

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        self.samples = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name

            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    label = self.class_to_idx[class_name]
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label