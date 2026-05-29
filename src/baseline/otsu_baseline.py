from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def convert_to_binary_label(label):
    label = int(label)

    if label == 2:
        return "no_tumor"

    return "tumor"


def apply_otsu_and_save(image_path, save_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, otsu_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    cv2.imwrite(str(save_path), otsu_mask)

    return image, blurred, otsu_mask


def extract_otsu_features(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    white_pixel_ratio = np.sum(mask == 255) / mask.size

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)
    else:
        contour_area = 0
        contour_perimeter = 0

    edges = cv2.Canny(blurred, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()

    features = {
        "white_pixel_ratio": white_pixel_ratio,
        "contour_area": contour_area,
        "contour_perimeter": contour_perimeter,
        "edge_density": edge_density,
    }

    for i, value in enumerate(hu_moments):
        features[f"hu_moment_{i + 1}"] = value

    return features


def rule_based_classifier(features):
    if (
        features["white_pixel_ratio"] > 0.25
        and features["contour_area"] > 3000
        and features["edge_density"] > 0.035
    ):
        return "tumor"

    return "no_tumor"


def save_pipeline_visualization(
    original,
    blurred,
    otsu_mask,
    true_label,
    pred_label,
    save_path,
):
    edges = cv2.Canny(blurred, 100, 200)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Preprocessed Image")
    axes[0].axis("off")

    axes[1].imshow(blurred, cmap="gray")
    axes[1].set_title("Gaussian Blur")
    axes[1].axis("off")

    axes[2].imshow(otsu_mask, cmap="gray")
    axes[2].set_title("Otsu Threshold")
    axes[2].axis("off")

    axes[3].imshow(edges, cmap="gray")
    axes[3].set_title("Canny Edges")
    axes[3].axis("off")

    fig.suptitle(f"True: {true_label} | Pred: {pred_label}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_confusion_matrix_visual(cm, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(cm)
    ax.set_title("Otsu Baseline Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    labels = ["no_tumor", "tumor"]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_otsu_baseline(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    df["label"] = df["label"].apply(convert_to_binary_label)

    print("Binary labels:")
    print(df["label"].value_counts())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = output_dir / "otsu_masks"
    visual_dir = output_dir / "visualizations"

    mask_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    y_true = []
    y_pred = []
    all_rows = []
    error_rows = []

    start_time = time.time()

    for idx, row in df.iterrows():
        image_path = row["image_path"]
        true_label = row["label"]

        mask_save_path = mask_dir / f"mask_{idx}.png"

        original, blurred, otsu_mask = apply_otsu_and_save(
            image_path=image_path,
            save_path=mask_save_path,
        )

        features = extract_otsu_features(image_path)
        pred_label = rule_based_classifier(features)

        y_true.append(true_label)
        y_pred.append(pred_label)

        result_row = {
            "image_path": image_path,
            "otsu_mask_path": str(mask_save_path),
            "true_label": true_label,
            "predicted_label": pred_label,
            "correct": true_label == pred_label,
            **features,
        }

        all_rows.append(result_row)

        if true_label != pred_label:
            error_rows.append(result_row)

        if idx < 30:
            save_pipeline_visualization(
                original=original,
                blurred=blurred,
                otsu_mask=otsu_mask,
                true_label=true_label,
                pred_label=pred_label,
                save_path=visual_dir / f"pipeline_{idx}.png",
            )

    print("\nPrediction distribution:")
    print(pd.Series(y_pred).value_counts())

    total_time = time.time() - start_time

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["no_tumor", "tumor"],
    )

    results_df = pd.DataFrame(all_rows)
    errors_df = pd.DataFrame(error_rows)

    results_df.to_csv(
        output_dir / "otsu_predictions.csv",
        index=False,
    )

    errors_df.to_csv(
        output_dir / "otsu_failed_examples.csv",
        index=False,
    )

    results_df.describe().to_csv(
        output_dir / "otsu_feature_summary.csv",
    )

    save_confusion_matrix_visual(
        cm=cm,
        output_path=output_dir / "confusion_matrix.png",
    )

    with open(output_dir / "otsu_metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"Total inference time: {total_time:.4f} seconds\n")
        f.write(
            f"Average inference time per image: "
            f"{total_time / len(df):.6f} seconds\n\n"
        )

        f.write("Confusion Matrix:\n")
        f.write(str(cm))

        f.write("\n\nClassification Report:\n")
        f.write(
            classification_report(
                y_true,
                y_pred,
                labels=["no_tumor", "tumor"],
                zero_division=0,
            )
        )

    print("\nOtsu baseline completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    run_otsu_baseline(
        csv_path="data/processed/test_preprocessed.csv",
        output_dir="results/otsu_baseline",
    )