"""
Complete preprocessing and augmentation pipeline for TN_2023 ultrasound dataset.

Steps:
  1. Read & clean TN_Annotations.xlsx
  2. Preprocess images (hair/artifact removal via blackhat morphology + inpainting)
  3. Augment images (histogram eq, CLAHE, Gaussian blur, edge enhancement)
  4. Save updated labels (original + augmented) as Excel and CSV
  5. Separate per-biomarker label files
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_IMAGE_DIR = os.path.join(BASE_DIR, "TN_2023")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "TN_Annotations.xlsx")

PREPROCESSED_DIR = os.path.join(BASE_DIR, "dataset_preprocessed")
AUGMENTED_DIR = os.path.join(BASE_DIR, "dataset_augmented")

OUTPUT_LABELS_CSV = os.path.join(BASE_DIR, "augmented_labels.csv")
OUTPUT_LABELS_XLSX = os.path.join(BASE_DIR, "augmented_labels.xlsx")
CLEANED_LABELS_CSV = os.path.join(BASE_DIR, "labels.csv")
BIOMARKER_DIR = os.path.join(BASE_DIR, "biomarker_labels")

BIOMARKERS = ["Zoom", "Sagital", "Neutral", "Caliper"]

ENHANCEMENT_TECHNIQUES = [
    "histogram_eq",
    "clahe",
    "gaussian_blur",
    "edge_enhancement",
]


# ──────────────────────────────────────────────
# STEP 1 — Read & clean annotations
# ──────────────────────────────────────────────
def load_and_clean_annotations(filepath: str) -> pd.DataFrame:
    """Load TN_Annotations.xlsx, drop junk columns/rows, cast types."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading & cleaning annotations")
    print("=" * 60)

    df = pd.read_excel(filepath)

    # Keep only the relevant columns
    df = df[["Image ID", "Zoom", "Sagital", "Neutral", "Caliper"]].copy()

    # Drop rows where Image ID is NaN (summary row at bottom)
    df = df.dropna(subset=["Image ID"])

    # Cast Image ID to int
    df["Image ID"] = df["Image ID"].astype(int)

    # Ensure biomarker columns are int (0/1)
    for col in BIOMARKERS:
        df[col] = df[col].astype(int)

    # Drop any rows where Image ID looks like a summary (e.g. > 300 or == counts)
    df = df[df["Image ID"] <= 205].reset_index(drop=True)

    print(f"  Cleaned annotations: {len(df)} images")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Class distributions:")
    for col in BIOMARKERS:
        counts = df[col].value_counts().to_dict()
        print(f"    {col}: {counts}")

    # Save cleaned labels
    df.to_csv(CLEANED_LABELS_CSV, index=False)
    print(f"  Saved cleaned labels to: {CLEANED_LABELS_CSV}")

    return df


# ──────────────────────────────────────────────
# STEP 2 — Preprocess images
# ──────────────────────────────────────────────
def preprocess_images(input_folder: str, output_folder: str, df: pd.DataFrame):
    """
    Preprocess each image:
      - Read from TN_2023/image_{id}.jpg
      - Apply blackhat morphology to detect hair/thin artifacts
      - Inpaint detected artifacts
      - Save as {id}.jpg in output folder
    """
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing images (artifact removal)")
    print("=" * 60)

    os.makedirs(output_folder, exist_ok=True)

    image_ids = df["Image ID"].tolist()
    success_count = 0
    fail_count = 0

    for img_id in tqdm(image_ids, desc="Preprocessing"):
        src_path = os.path.join(input_folder, f"image_{img_id}.jpg")

        if not os.path.exists(src_path):
            print(f"  WARNING: {src_path} not found, skipping")
            fail_count += 1
            continue

        image = cv2.imread(src_path)
        if image is None:
            print(f"  WARNING: Could not read {src_path}, skipping")
            fail_count += 1
            continue

        # Convert to grayscale for morphological operations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blackhat transform to detect thin dark structures (hair, artifacts)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Threshold the blackhat result
        _, binary = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # Dilate to connect nearby regions
        dilated = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)

        # Inpaint the detected artifact regions
        inpainted = cv2.inpaint(image, dilated, 3, cv2.INPAINT_TELEA)

        # Save with numeric-only filename
        out_path = os.path.join(output_folder, f"{img_id}.jpg")
        cv2.imwrite(out_path, inpainted)
        success_count += 1

    print(f"  Preprocessed: {success_count} images, Failed: {fail_count}")
    print(f"  Output folder: {output_folder}")


# ──────────────────────────────────────────────
# STEP 3 — Augment images
# ──────────────────────────────────────────────
def apply_histogram_eq(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else equalized


def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else enhanced


def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_edge_enhancement(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 100, 200)
    if len(image.shape) == 3:
        color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 1, color_edges, 0.5, 0)
    else:
        return cv2.addWeighted(gray, 0.7, edges, 0.3, 0)


TECHNIQUE_MAP = {
    "histogram_eq": apply_histogram_eq,
    "clahe": apply_clahe,
    "gaussian_blur": apply_gaussian_blur,
    "edge_enhancement": apply_edge_enhancement,
}


def augment_images(
    preprocessed_folder: str,
    output_folder: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each preprocessed image:
      - Copy the original into output folder
      - Apply each enhancement technique and save with a new ID
      - Build a new label DataFrame including augmented entries

    Returns the combined DataFrame (original + augmented).
    """
    print("\n" + "=" * 60)
    print("STEP 3: Augmenting images")
    print("=" * 60)

    os.makedirs(output_folder, exist_ok=True)

    max_id = df["Image ID"].max()
    next_id = max_id + 1

    augmented_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
        img_id = int(row["Image ID"])
        img_path = os.path.join(preprocessed_folder, f"{img_id}.jpg")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  WARNING: Could not read {img_path}, skipping")
            continue

        labels = {b: int(row[b]) for b in BIOMARKERS}

        # Save original (copy into augmented folder)
        cv2.imwrite(os.path.join(output_folder, f"{img_id}.jpg"), image)
        augmented_rows.append(
            {"Image ID": img_id, **labels, "Augmentation": "original"}
        )

        # Apply each enhancement
        for tech_name in ENHANCEMENT_TECHNIQUES:
            enhanced = TECHNIQUE_MAP[tech_name](image.copy())
            new_filename = f"{next_id}.jpg"
            cv2.imwrite(os.path.join(output_folder, new_filename), enhanced)
            augmented_rows.append(
                {"Image ID": next_id, **labels, "Augmentation": tech_name}
            )
            next_id += 1

    augmented_df = pd.DataFrame(augmented_rows)

    # Reorder columns
    augmented_df = augmented_df[
        ["Image ID", "Zoom", "Sagital", "Neutral", "Caliper", "Augmentation"]
    ]

    total_original = len(df)
    total_augmented = len(augmented_df) - total_original

    print(f"  Original images: {total_original}")
    print(f"  Augmented images: {total_augmented}")
    print(f"  Total images: {len(augmented_df)}")
    print(f"  Techniques applied: {ENHANCEMENT_TECHNIQUES}")

    return augmented_df


# ──────────────────────────────────────────────
# STEP 4 — Save updated labels
# ──────────────────────────────────────────────
def save_labels(augmented_df: pd.DataFrame):
    """Save the combined labels as both CSV and Excel."""
    print("\n" + "=" * 60)
    print("STEP 4: Saving updated labels")
    print("=" * 60)

    # Save full augmented labels (with augmentation type column)
    augmented_df.to_csv(OUTPUT_LABELS_CSV, index=False)
    augmented_df.to_excel(OUTPUT_LABELS_XLSX, index=False)
    print(f"  Saved CSV:  {OUTPUT_LABELS_CSV}")
    print(f"  Saved XLSX: {OUTPUT_LABELS_XLSX}")

    # Also save a version without the Augmentation column for downstream compatibility
    compat_df = augmented_df[["Image ID", "Zoom", "Sagital", "Neutral", "Caliper"]]
    compat_csv = os.path.join(BASE_DIR, "augmented_labels_compat.csv")
    compat_df.to_csv(compat_csv, index=False)
    print(f"  Saved compatible CSV (no augmentation column): {compat_csv}")


# ──────────────────────────────────────────────
# STEP 5 — Separate per-biomarker labels
# ──────────────────────────────────────────────
def separate_biomarker_labels(augmented_df: pd.DataFrame):
    """Create a separate CSV for each biomarker with columns [Image ID, <biomarker>]."""
    print("\n" + "=" * 60)
    print("STEP 5: Separating per-biomarker labels")
    print("=" * 60)

    os.makedirs(BIOMARKER_DIR, exist_ok=True)

    for biomarker in BIOMARKERS:
        bio_df = augmented_df[["Image ID", biomarker]].copy()
        out_path = os.path.join(BIOMARKER_DIR, f"{biomarker}.csv")
        bio_df.to_csv(out_path, index=False)

        # Also save in root for compatibility with main.py
        root_path = os.path.join(BASE_DIR, f"{biomarker}.csv")
        bio_df.to_csv(root_path, index=False)

        counts = bio_df[biomarker].value_counts().to_dict()
        print(f"  {biomarker}: saved {len(bio_df)} rows -> {out_path}")
        print(f"    Distribution: {counts}")

    print(f"\n  All biomarker label files saved in: {BIOMARKER_DIR}/")
    print(f"  Also saved in project root for main.py compatibility")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TN_2023 Preprocessing & Augmentation Pipeline")
    print("=" * 60)

    # Step 1: Clean annotations
    df = load_and_clean_annotations(ANNOTATIONS_FILE)

    # Step 2: Preprocess images
    preprocess_images(RAW_IMAGE_DIR, PREPROCESSED_DIR, df)

    # Step 3: Augment images
    augmented_df = augment_images(PREPROCESSED_DIR, AUGMENTED_DIR, df)

    # Step 4: Save updated labels
    save_labels(augmented_df)

    # Step 5: Separate biomarker labels
    separate_biomarker_labels(augmented_df)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Summary:")
    print(f"    Raw images:          {RAW_IMAGE_DIR} ({len(df)} images)")
    print(f"    Preprocessed:        {PREPROCESSED_DIR}")
    print(f"    Augmented:           {AUGMENTED_DIR} ({len(augmented_df)} images)")
    print(f"    Labels (CSV):        {OUTPUT_LABELS_CSV}")
    print(f"    Labels (XLSX):       {OUTPUT_LABELS_XLSX}")
    print(f"    Biomarker labels:    {BIOMARKER_DIR}/")
    for b in BIOMARKERS:
        print(f"      - {b}.csv")


if __name__ == "__main__":
    main()
