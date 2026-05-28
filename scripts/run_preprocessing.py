"""
Preprocessing & augmentation pipeline for the TN_2023 ultrasound dataset.

Steps:
  1. Read & clean TN_Annotations.xlsx
  2. Preprocess images (artifact removal via blackhat morphology + inpainting)
  3. Augment images (histogram eq, CLAHE, Gaussian blur, edge enhancement)
  4. Save updated labels as CSV and Excel
  5. Write per-biomarker label CSVs

Run standalone:
    python scripts/run_preprocessing.py
Or import the main() function from main.py.
"""

import sys
import os

# Ensure project root is importable when this script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config

# ── Augmentation technique registry ──────────────────────────────────────────
ENHANCEMENT_TECHNIQUES = ["histogram_eq", "clahe", "gaussian_blur", "edge_enhancement"]


def _apply_histogram_eq(image):
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    eq       = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else eq


def _apply_clahe(image):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else enhanced


def _apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def _apply_edge_enhancement(image):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 100, 200)
    if image.ndim == 3:
        return cv2.addWeighted(image, 1, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.5, 0)
    return cv2.addWeighted(gray, 0.7, edges, 0.3, 0)


_TECHNIQUE_MAP = {
    "histogram_eq":     _apply_histogram_eq,
    "clahe":            _apply_clahe,
    "gaussian_blur":    _apply_gaussian_blur,
    "edge_enhancement": _apply_edge_enhancement,
}


# ── Step 1 ────────────────────────────────────────────────────────────────────
def load_and_clean_annotations() -> pd.DataFrame:
    """Load TN_Annotations.xlsx, drop junk columns/rows, cast types."""
    print("\n" + "=" * 60)
    print("STEP 1 — Loading & cleaning annotations")
    print("=" * 60)

    df = pd.read_excel(config.ANNOTATIONS_FILE)
    df = df[["Image ID"] + config.BIOMARKERS].copy()
    df = df.dropna(subset=["Image ID"])
    df["Image ID"] = df["Image ID"].astype(int)
    for col in config.BIOMARKERS:
        df[col] = df[col].astype(int)
    df = df[df["Image ID"] <= 205].reset_index(drop=True)

    print(f"  Cleaned annotations: {len(df)} images")
    for col in config.BIOMARKERS:
        print(f"    {col}: {df[col].value_counts().to_dict()}")

    os.makedirs(config.LABELS_DIR, exist_ok=True)
    df.to_csv(config.LABELS_CSV, index=False)
    print(f"  Saved cleaned labels → {config.LABELS_CSV}")
    return df


# ── Step 2 ────────────────────────────────────────────────────────────────────
def preprocess_images(df: pd.DataFrame) -> None:
    """Blackhat morphology + inpainting to remove hair/text artifacts."""
    print("\n" + "=" * 60)
    print("STEP 2 — Preprocessing images (artifact removal)")
    print("=" * 60)

    os.makedirs(config.PREPROCESSED_DIR, exist_ok=True)
    ok = fail = 0

    for img_id in tqdm(df["Image ID"].tolist(), desc="Preprocessing"):
        src = os.path.join(config.RAW_DIR, f"image_{img_id}.jpg")
        if not os.path.exists(src):
            print(f"  WARNING: {src} not found – skipping")
            fail += 1
            continue

        image = cv2.imread(src)
        if image is None:
            print(f"  WARNING: could not read {src} – skipping")
            fail += 1
            continue

        gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, binary  = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        dilated    = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)
        inpainted  = cv2.inpaint(image, dilated, 3, cv2.INPAINT_TELEA)

        cv2.imwrite(os.path.join(config.PREPROCESSED_DIR, f"{img_id}.jpg"), inpainted)
        ok += 1

    print(f"  Preprocessed: {ok}  |  Failed: {fail}")
    print(f"  Output → {config.PREPROCESSED_DIR}")


# ── Step 3 ────────────────────────────────────────────────────────────────────
def augment_images(df: pd.DataFrame) -> pd.DataFrame:
    """Copy originals + apply 4 enhancement techniques for every image."""
    print("\n" + "=" * 60)
    print("STEP 3 — Augmenting images")
    print("=" * 60)

    os.makedirs(config.AUGMENTED_DIR, exist_ok=True)

    next_id = int(df["Image ID"].max()) + 1
    rows    = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        img_id   = int(row["Image ID"])
        src      = os.path.join(config.PREPROCESSED_DIR, f"{img_id}.jpg")
        image    = cv2.imread(src)
        if image is None:
            print(f"  WARNING: could not read {src} – skipping")
            continue

        labels = {b: int(row[b]) for b in config.BIOMARKERS}

        # Save original copy
        cv2.imwrite(os.path.join(config.AUGMENTED_DIR, f"{img_id}.jpg"), image)
        rows.append({"Image ID": img_id, **labels, "Augmentation": "original"})

        # Apply each technique
        for name in ENHANCEMENT_TECHNIQUES:
            enhanced = _TECHNIQUE_MAP[name](image.copy())
            cv2.imwrite(os.path.join(config.AUGMENTED_DIR, f"{next_id}.jpg"), enhanced)
            rows.append({"Image ID": next_id, **labels, "Augmentation": name})
            next_id += 1

    aug_df = pd.DataFrame(rows)[["Image ID"] + config.BIOMARKERS + ["Augmentation"]]
    print(f"  Original: {len(df)}  |  Augmented: {len(aug_df) - len(df)}  |  Total: {len(aug_df)}")
    return aug_df


# ── Step 4 ────────────────────────────────────────────────────────────────────
def save_labels(aug_df: pd.DataFrame) -> None:
    """Persist the combined label DataFrame as CSV and Excel."""
    print("\n" + "=" * 60)
    print("STEP 4 — Saving labels")
    print("=" * 60)

    os.makedirs(config.LABELS_DIR, exist_ok=True)
    aug_df.to_csv(config.AUGMENTED_LABELS_CSV,  index=False)
    aug_df.to_excel(config.AUGMENTED_LABELS_XLSX, index=False)
    print(f"  CSV  → {config.AUGMENTED_LABELS_CSV}")
    print(f"  XLSX → {config.AUGMENTED_LABELS_XLSX}")

    # Compatibility copy without the Augmentation column
    compat = aug_df[["Image ID"] + config.BIOMARKERS]
    compat_path = os.path.join(config.LABELS_DIR, "augmented_labels_compat.csv")
    compat.to_csv(compat_path, index=False)
    print(f"  Compat CSV → {compat_path}")


# ── Step 5 ────────────────────────────────────────────────────────────────────
def separate_biomarker_labels(aug_df: pd.DataFrame) -> None:
    """Write one [Image ID, <biomarker>] CSV per biomarker."""
    print("\n" + "=" * 60)
    print("STEP 5 — Separating per-biomarker labels")
    print("=" * 60)

    os.makedirs(config.LABELS_DIR, exist_ok=True)
    for biomarker in config.BIOMARKERS:
        out = config.biomarker_csv(biomarker)
        aug_df[["Image ID", biomarker]].to_csv(out, index=False)
        counts = aug_df[biomarker].value_counts().to_dict()
        print(f"  {biomarker}: {len(aug_df)} rows → {out}  {counts}")


# ── Orchestrator ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  TN_2023 Preprocessing & Augmentation Pipeline")
    print("=" * 60)

    df     = load_and_clean_annotations()
    preprocess_images(df)
    aug_df = augment_images(df)
    save_labels(aug_df)
    separate_biomarker_labels(aug_df)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Raw images     : {config.RAW_DIR}")
    print(f"  Preprocessed   : {config.PREPROCESSED_DIR}")
    print(f"  Augmented      : {config.AUGMENTED_DIR}  ({len(aug_df)} images)")
    print(f"  Labels         : {config.LABELS_DIR}")


if __name__ == "__main__":
    main()
