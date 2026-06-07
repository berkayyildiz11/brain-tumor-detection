"""
src/classical_ml/hog_svm.py

Brain Tumor MRI — HOG + SVM Pipeline
=====================================================================
Proje yapısına tam uyumlu:
  Veri   → data/processed/images/{train,val,test}/{class}/
  CSV    → data/processed/{train,val,test}_preprocessed.csv
  Çıktı  → results/classical_ml/hog_svm/
=====================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PROJE KOKU
# Bu dosya src/classical_ml/hog_svm.py konumunda → iki üst dizine çık.
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────────────────
# KONFİGÜRASYON — gerekirse buradan düzenle
# ─────────────────────────────────────────────────────────────────────
CFG = {
    # Veri dizinleri
    "data_root": PROJECT_ROOT / "data" / "processed" / "images",
    "csv_root":  PROJECT_ROOT / "data" / "processed",

    # Çıktı dizini
    "out_dir":   PROJECT_ROOT / "results" / "classical_ml" / "hog_svm",

    # Görüntü yeniden ölçekleme boyutu
    "img_size": (128, 128),

    # HOG parametreleri
    "hog_orientations":    9,
    "hog_pixels_per_cell": (8, 8),
    "hog_cells_per_block": (2, 2),
    "hog_block_norm":      "L2-Hys",

    # SVM — use_grid_search=True ise aşağıdaki değerler yok sayılır
    "svm_C":      10,
    "svm_kernel": "rbf",
    "svm_gamma":  "scale",

    # GridSearchCV: True yaparsan en iyi C/gamma/kernel aranır (yavaş)
    "use_grid_search": False,

    # Sınıf adları (sabit sıra → tüm raporlarda tutarlılık)
    "classes": ["glioma", "meningioma", "notumor", "pituitary"],
}


# ─────────────────────────────────────────────────────────────────────
# LOGGER — hem ekrana hem dosyaya yazar
# ─────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", encoding="utf-8")

    def log(self, msg: str = ""):
        print(msg)
        self._f.write(msg + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


# ─────────────────────────────────────────────────────────────────────
# 1. VERİ YÜKLEME
# Önce CSV metadata dosyasına bakar; yoksa klasör yapısından yükler.
# ─────────────────────────────────────────────────────────────────────
def load_split(split: str, cfg: dict) -> tuple:
    """
    Returns: (images: np.ndarray [N,H,W,3], labels: list[str])
    """
    csv_path = Path(cfg["csv_root"]) / f"{split}_preprocessed.csv"
    if csv_path.exists():
        return _load_from_csv(csv_path, cfg)
    folder = Path(cfg["data_root"]) / split
    return _load_from_folder(folder, cfg)


def _load_from_csv(csv_path: Path, cfg: dict) -> tuple:
    df = pd.read_csv(csv_path)

    if "label_name" in df.columns:
        df["label"] = df["label_name"]

    # Sütun adlarını normalize et
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("image_path", "filepath", "path", "file_path", "filename"):
            col_map[c] = "image_path"
        elif lc in ("label", "class", "category", "target"):
            col_map[c] = "label"
    df.rename(columns=col_map, inplace=True)

    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{csv_path.name} içinde 'image_path' ve 'label' sütunları bulunamadı.\n"
            f"Mevcut sütunlar: {list(df.columns)}"
        )

    images, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  CSV: {csv_path.stem}", leave=True):
        img_path = Path(str(row["image_path"]))
        if not img_path.is_absolute():
            img_path = Path(cfg["csv_root"]).parents[1] / img_path

        try:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.float32) / 255.0
            img = resize(img, cfg["img_size"], anti_aliasing=True)
            images.append(img)
            labels.append(str(row["label"]).lower().strip())
        except Exception as e:
            print(f"    [UYARI] {img_path.name} atlandı: {e}")

    return np.array(images, dtype=np.float32), labels


def _load_from_folder(folder: Path, cfg: dict) -> tuple:
    images, labels = [], []
    class_dirs = sorted([d for d in folder.iterdir() if d.is_dir()])

    for cls_dir in class_dirs:
        files = (list(cls_dir.glob("*.jpg"))
                 + list(cls_dir.glob("*.jpeg"))
                 + list(cls_dir.glob("*.png")))
        for fp in tqdm(files, desc=f"    {cls_dir.name}", leave=False):
            try:
                img = Image.open(fp).convert("RGB")
                img = np.array(img, dtype=np.float32) / 255.0
                img = resize(img, cfg["img_size"], anti_aliasing=True)
                images.append(img)
                labels.append(cls_dir.name.lower().strip())
            except Exception as e:
                print(f"    [UYARI] {fp.name} atlandı: {e}")

    return np.array(images, dtype=np.float32), labels


# ─────────────────────────────────────────────────────────────────────
# 2. HOG FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────
def extract_hog(images: np.ndarray, cfg: dict, desc: str = "HOG") -> np.ndarray:
    feats = []
    for img in tqdm(images, desc=f"  {desc}", leave=True):
        gray = rgb2gray(img)
        f = hog(
            gray,
            orientations=cfg["hog_orientations"],
            pixels_per_cell=cfg["hog_pixels_per_cell"],
            cells_per_block=cfg["hog_cells_per_block"],
            block_norm=cfg["hog_block_norm"],
            feature_vector=True,
        )
        feats.append(f)
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# 3. GÖRSELLEŞTİRMELER
# ─────────────────────────────────────────────────────────────────────
def save_hog_grid(train_imgs: np.ndarray, train_labels: list,
                  cfg: dict, path: Path):
    """Her sınıftan 1 örnek → orijinal + HOG feature map."""
    classes = cfg["classes"]
    fig, axes = plt.subplots(len(classes), 2, figsize=(8, 4 * len(classes)))

    for i, cls in enumerate(classes):
        idx = next((j for j, l in enumerate(train_labels) if l == cls), 0)
        img  = train_imgs[idx]
        gray = rgb2gray(img)
        _, hog_img = hog(
            gray,
            orientations=cfg["hog_orientations"],
            pixels_per_cell=cfg["hog_pixels_per_cell"],
            cells_per_block=cfg["hog_cells_per_block"],
            block_norm=cfg["hog_block_norm"],
            visualize=True,
            feature_vector=True,
        )
        axes[i, 0].imshow(img);          axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{cls} — Orijinal", fontsize=10)
        axes[i, 1].imshow(hog_img, cmap="inferno"); axes[i, 1].axis("off")
        axes[i, 1].set_title(f"{cls} — HOG", fontsize=10)

    plt.suptitle("HOG Feature Map — Her Sınıftan 1 Örnek",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_class_distribution(label_dict: dict, path: Path):
    splits  = list(label_dict.keys())
    palette = sns.color_palette("Set2", 4)

    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4))
    if len(splits) == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        labels  = label_dict[split]
        unique, counts = np.unique(labels, return_counts=True)
        bars = ax.bar(unique, counts,
                      color=[palette[i % len(palette)] for i in range(len(unique))])
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title(f"{split}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Sınıf");  ax.set_ylabel("Görüntü Sayısı")
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, max(counts) * 1.18)

    plt.suptitle("Veri Seti Sınıf Dağılımı", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true, y_pred, classes: list,
                           title: str, path: Path):
    cm   = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=20)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_per_class_bar(report_dict: dict, classes: list,
                       title: str, path: Path):
    metrics = ["precision", "recall", "f1-score"]
    colors  = ["#4C72B0", "#55A868", "#C44E52"]
    x       = np.arange(len(classes))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [report_dict.get(cls, {}).get(m, 0) for cls in classes]
        ax.bar(x + i * width, vals, width, label=m.capitalize(), color=c)

    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Skor")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# 4. MODEL EĞİTİMİ
# ─────────────────────────────────────────────────────────────────────
def train_svm(X_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> SVC:
    if cfg["use_grid_search"]:
        print("  GridSearchCV başlatıldı (birkaç dakika sürebilir)...")
        param_grid = {
            "C":      [0.1, 1, 10, 100],
            "gamma":  ["scale", "auto", 0.001, 0.01],
            "kernel": ["rbf", "linear"],
        }
        gs = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid, cv=3, n_jobs=-1, verbose=1, scoring="accuracy",
        )
        gs.fit(X_train, y_train)
        print(f"  En iyi parametreler : {gs.best_params_}")
        print(f"  CV accuracy         : {gs.best_score_:.4f}")
        return gs.best_estimator_

    svm = SVC(
        C=cfg["svm_C"],
        kernel=cfg["svm_kernel"],
        gamma=cfg["svm_gamma"],
        probability=False,
        random_state=42,
    )
    svm.fit(X_train, y_train)
    return svm


# ─────────────────────────────────────────────────────────────────────
# 5. DEĞERLENDİRME
# ─────────────────────────────────────────────────────────────────────
def evaluate_split(model: SVC, scaler: StandardScaler, le: LabelEncoder,
                   X_raw: np.ndarray, labels: list,
                   split: str, cfg: dict, out: Path, log: Logger) -> dict:

    X_sc       = scaler.transform(X_raw)
    y_enc      = le.transform(labels)
    y_pred_enc = model.predict(X_sc)

    y_true_str = np.array(labels)
    y_pred_str = le.inverse_transform(y_pred_enc)

    acc        = accuracy_score(y_true_str, y_pred_str)
    report_txt = classification_report(y_true_str, y_pred_str,
                                       target_names=cfg["classes"])
    report_dct = classification_report(y_true_str, y_pred_str,
                                       target_names=cfg["classes"],
                                       output_dict=True)

    # Macro ROC-AUC (OvR)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_sc)
        else:
            probs = model.decision_function(X_sc)
        auc = roc_auc_score(y_enc, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    log.log(f"\n{'─'*58}")
    log.log(f"  {split.upper()} SONUÇLARI")
    log.log(f"{'─'*58}")
    log.log(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    log.log(f"  Macro AUC : {auc:.4f}")
    log.log(f"\n{report_txt}")

    # Görseller
    save_confusion_matrix(
        y_true_str, y_pred_str, cfg["classes"],
        f"Confusion Matrix — {split}",
        out / f"confusion_matrix_{split.lower()}.png",
    )
    save_per_class_bar(
        report_dct, cfg["classes"],
        f"Per-Class Metrics — {split}",
        out / f"per_class_metrics_{split.lower()}.png",
    )

    # Tahmin CSV
    pd.DataFrame({
        "true": y_true_str,
        "predicted": y_pred_str,
        "correct": y_true_str == y_pred_str,
    }).to_csv(out / f"predictions_{split.lower()}.csv", index=False)

    return {"split": split, "accuracy": acc, "macro_auc": auc,
            "report": report_dct}


# ─────────────────────────────────────────────────────────────────────
# 6. ANA PIPELINE
# ─────────────────────────────────────────────────────────────────────
def main():
    out = Path(CFG["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "hog_svm_log.txt")

    log.log("=" * 58)
    log.log("  Brain Tumor MRI — HOG + SVM Pipeline")
    log.log("=" * 58)
    log.log(f"  Proje kökü   : {PROJECT_ROOT}")
    log.log(f"  Çıktı dizini : {out}")
    log.log(
        f"\n  HOG  → orientations={CFG['hog_orientations']}, "
        f"pixels_per_cell={CFG['hog_pixels_per_cell']}, "
        f"cells_per_block={CFG['hog_cells_per_block']}"
    )
    log.log(
        f"  SVM  → kernel={CFG['svm_kernel']}, "
        f"C={CFG['svm_C']}, gamma={CFG['svm_gamma']}"
    )

    # ── 1. Veri yükleme ───────────────────────────────────────────────
    log.log("\n[1/5] Veri yükleniyor...")
    splits_raw: dict[str, tuple] = {}
    label_dict: dict[str, list]  = {}

    for split in ("train", "val", "test"):
        log.log(f"\n  ▶ {split}")
        imgs, labels = load_split(split, CFG)
        splits_raw[split] = (imgs, labels)
        label_dict[split] = labels
        unique, counts = np.unique(labels, return_counts=True)
        for cls, cnt in zip(unique, counts):
            log.log(f"    {cls:15s}: {cnt}")
        log.log(f"    {'TOPLAM':15s}: {len(labels)}")

    # ── 2. Dağılım grafiği ────────────────────────────────────────────
    save_class_distribution(label_dict, out / "class_distribution.png")
    log.log("\n  Sınıf dağılımı grafiği kaydedildi.")

    # ── 3. HOG çıkarımı ───────────────────────────────────────────────
    log.log("\n[2/5] HOG feature extraction...")
    feats: dict[str, np.ndarray] = {}
    for split, (imgs, _) in splits_raw.items():
        feats[split] = extract_hog(imgs, CFG, desc=f"HOG [{split}]")
    log.log(f"  Feature vektör boyutu: {feats['train'].shape[1]}")

    # HOG görselleştirme (her sınıftan 1 örnek)
    log.log("  HOG görselleştirmesi kaydediliyor...")
    save_hog_grid(
        splits_raw["train"][0], splits_raw["train"][1],
        CFG, out / "hog_visualization.png",
    )

    # ── 4. Label encoding & ölçekleme ─────────────────────────────────
    log.log("\n[3/5] Label encoding & StandardScaler...")
    le = LabelEncoder()
    le.fit(CFG["classes"])

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(feats["train"])
    y_train    = le.transform(splits_raw["train"][1])

    # ── 5. Eğitim ─────────────────────────────────────────────────────
    log.log("\n[4/5] SVM eğitimi...")
    t0    = time.time()
    model = train_svm(X_train_sc, y_train, CFG)
    log.log(f"  Eğitim süresi: {time.time() - t0:.1f} sn")

    # ── 6. Değerlendirme ──────────────────────────────────────────────
    log.log("\n[5/5] Değerlendirme...")
    results = []
    for split in ("train", "val", "test"):
        imgs, labels = splits_raw[split]
        r = evaluate_split(
            model, scaler, le,
            feats[split], labels,
            split, CFG, out, log,
        )
        results.append(r)

    # ── Özet ──────────────────────────────────────────────────────────
    log.log("\n" + "=" * 58)
    log.log("  ÖZET TABLO")
    log.log("=" * 58)
    log.log(f"  {'Split':<12} {'Accuracy':>12} {'Macro AUC':>12}")
    log.log(f"  {'-'*38}")
    for r in results:
        log.log(
            f"  {r['split']:<12} {r['accuracy']*100:>11.2f}%"
            f" {r['macro_auc']:>12.4f}"
        )
    log.log("=" * 58)

    # Özet CSV
    pd.DataFrame([
        {"split": r["split"],
         "accuracy": r["accuracy"],
         "macro_auc": r["macro_auc"]}
        for r in results
    ]).to_csv(out / "summary_metrics.csv", index=False)

    # Artefaktları kaydet
    joblib.dump(model,  out / "svm_model.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    joblib.dump(le,     out / "label_encoder.pkl")
    log.log(f"\n  Artefaktlar kaydedildi → {out}/")

    log.close()
    print(f"\nTamamlandı. Tüm çıktılar: {out}")


if __name__ == "__main__":
    main()
