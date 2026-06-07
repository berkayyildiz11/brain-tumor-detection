from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tqdm import tqdm


def build_hog_descriptor(
    image_size: int,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
) -> cv2.HOGDescriptor:
    win_size = (image_size, image_size)
    block_size = (
        pixels_per_cell * cells_per_block,
        pixels_per_cell * cells_per_block,
    )
    block_stride = (pixels_per_cell, pixels_per_cell)
    cell_size = (pixels_per_cell, pixels_per_cell)

    return cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=orientations,
    )


def load_hog_features(
    csv_path: str | Path,
    image_size: int,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path).copy()
    descriptor = build_hog_descriptor(
        image_size=image_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )

    features: list[np.ndarray] = []
    labels: list[int] = []
    valid_rows: list[dict] = []

    for row in tqdm(
        df.to_dict(orient="records"),
        desc=f"Extracting HOG ({Path(csv_path).name})",
    ):
        image_path = row["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        resized = cv2.resize(image, (image_size, image_size))
        hog_vector = descriptor.compute(resized)

        if hog_vector is None:
            continue

        features.append(hog_vector.flatten())
        labels.append(int(row["label"]))
        valid_rows.append(row)

    if not features:
        raise ValueError(f"No valid image read from {csv_path}")

    return np.asarray(features), np.asarray(labels), pd.DataFrame(valid_rows)


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("HOG + SVM Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_split(
    model: Pipeline,
    split_name: str,
    x_data: np.ndarray,
    y_true: np.ndarray,
    split_df: pd.DataFrame,
    class_names: list[str],
    output_dir: Path,
) -> dict:
    start = time.time()
    y_pred = model.predict(x_data)
    elapsed = time.time() - start

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    pred_df = split_df.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred"] = y_pred
    pred_df["pred_label_name"] = pred_df["y_pred"].apply(
        lambda idx: class_names[int(idx)]
    )
    pred_df["correct"] = pred_df["y_true"] == pred_df["y_pred"]
    pred_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    errors_df = pred_df[pred_df["correct"] == False].copy()  # noqa: E712
    errors_df.to_csv(output_dir / f"{split_name}_errors.csv", index=False)

    save_confusion_matrix(
        cm=cm,
        class_names=class_names,
        out_path=output_dir / f"{split_name}_confusion_matrix.png",
    )

    return {
        "split": split_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "inference_time_sec": float(elapsed),
        "avg_inference_time_per_image_sec": float(elapsed / len(y_true)),
        "num_samples": int(len(y_true)),
        "num_errors": int((y_true != y_pred).sum()),
    }


def run_hog_svm(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    output_dir: str | Path,
    image_size: int = 224,
    orientations: int = 9,
    pixels_per_cell: int = 8,
    cells_per_block: int = 2,
    cv_folds: int = 3,
    n_jobs: int = 1,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    x_train, y_train, train_df = load_hog_features(
        csv_path=train_csv,
        image_size=image_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    x_val, y_val, val_df = load_hog_features(
        csv_path=val_csv,
        image_size=image_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    x_test, y_test, test_df = load_hog_features(
        csv_path=test_csv,
        image_size=image_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )

    class_names = (
        pd.concat([train_df, val_df, test_df], axis=0)
        .sort_values("label")["label_name"]
        .drop_duplicates()
        .tolist()
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=128, random_state=42)),
            ("svm", SVC()),
        ]
    )

    param_grid = {
        "svm__kernel": ["rbf", "linear"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", "auto"],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=n_jobs,
        scoring="f1_macro",
        verbose=2,
    )

    train_start = time.time()
    grid.fit(x_train, y_train)
    train_elapsed = time.time() - train_start

    best_model = grid.best_estimator_

    val_metrics = evaluate_split(
        model=best_model,
        split_name="validation",
        x_data=x_val,
        y_true=y_val,
        split_df=val_df,
        class_names=class_names,
        output_dir=output_path,
    )
    test_metrics = evaluate_split(
        model=best_model,
        split_name="test",
        x_data=x_test,
        y_true=y_test,
        split_df=test_df,
        class_names=class_names,
        output_dir=output_path,
    )

    run_report = {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "hog_config": {
            "image_size": image_size,
            "orientations": orientations,
            "pixels_per_cell": pixels_per_cell,
            "cells_per_block": cells_per_block,
        },
        "grid_search": {
            "cv_folds": cv_folds,
            "best_params": grid.best_params_,
            "best_cv_macro_f1": float(grid.best_score_),
            "num_candidates": int(len(grid.cv_results_["params"])),
        },
        "timing": {
            "training_time_sec": float(train_elapsed),
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
    }

    with open(output_path / "hog_svm_report.json", "w", encoding="utf-8") as fp:
        json.dump(run_report, fp, indent=2)

    cv_results = pd.DataFrame(grid.cv_results_).sort_values(
        "rank_test_score",
        ascending=True,
    )
    cv_results.to_csv(output_path / "grid_search_results.csv", index=False)

    print("HOG + SVM finished.")
    print(f"Best params: {grid.best_params_}")
    print(f"Validation macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Results saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOG + SVM for brain tumor detection")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/processed/train_preprocessed.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/processed/val_preprocessed.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/processed/test_preprocessed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/classical_ml/hog_svm",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--orientations", type=int, default=9)
    parser.add_argument("--pixels-per-cell", type=int, default=8)
    parser.add_argument("--cells-per-block", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of CPU cores to use. Set to 1 to avoid MemoryError.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hog_svm(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        image_size=args.image_size,
        orientations=args.orientations,
        pixels_per_cell=args.pixels_per_cell,
        cells_per_block=args.cells_per_block,
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
    )
