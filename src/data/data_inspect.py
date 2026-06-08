from pathlib import Path
from PIL import Image

DATASET_DIR = Path(
    "data/raw/brain-tumor-mri-data/versions/1/brain-tumor-mri-dataset"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def main():
    classes = sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])

    print("Classes:")
    for class_name in classes:
        image_paths = [
            p for p in (DATASET_DIR / class_name).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"{class_name}: {len(image_paths)} images")

        sample_image = image_paths[0]
        with Image.open(sample_image) as img:
            print(f"  sample: {sample_image.name}")
            print(f"  size: {img.size}")
            print(f"  mode: {img.mode}")

if __name__ == "__main__":
    main()