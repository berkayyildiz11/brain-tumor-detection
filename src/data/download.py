from pathlib import Path
import shutil

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "brain-tumor-mri-dataset"
EXPECTED_CLASSES = {"glioma", "meningioma", "notumor", "pituitary"}

TARGET_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brain-tumor-mri-data"
    / "versions"
    / "1"
    / DATASET_NAME
)


def has_expected_classes(path: Path) -> bool:
    return path.is_dir() and EXPECTED_CLASSES.issubset(
        {item.name.lower() for item in path.iterdir() if item.is_dir()}
    )


def find_downloaded_dataset(download_path: Path) -> Path:
    named_dataset_dir = download_path / DATASET_NAME
    if has_expected_classes(named_dataset_dir):
        return named_dataset_dir

    if has_expected_classes(download_path):
        return download_path

    for candidate in download_path.rglob(DATASET_NAME):
        if has_expected_classes(candidate):
            return candidate

    for candidate in download_path.rglob("*"):
        if has_expected_classes(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find the expected dataset folder in the Kaggle download. "
        f"Looked under: {download_path}"
    )


def copy_dataset_to_project(source_dir: Path, target_dir: Path) -> None:
    if source_dir.resolve() == target_dir.resolve():
        print(f"Dataset is already in the expected path: {target_dir}")
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    print(f"Dataset copied to: {target_dir}")


def main() -> None:
    print("Downloading dataset from Kaggle...")

    download_path = Path(
        kagglehub.dataset_download("tombackert/brain-tumor-mri-data")
    )
    print(f"Kaggle download path: {download_path}")

    dataset_path = find_downloaded_dataset(download_path)
    copy_dataset_to_project(dataset_path, TARGET_DATASET_DIR)

    print("Dataset ready at:")
    print(TARGET_DATASET_DIR)


if __name__ == "__main__":
    main()
