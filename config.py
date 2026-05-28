"""
Central configuration for the Quantitative-Evaluation-of-Explanation-Quality project.

Every path, constant, and hyperparameter that is shared across the
preprocessing, training, and evaluation stages lives here.
Import this module instead of hard-coding paths in individual scripts.
"""

import os

# ── Project root ─────────────────────────────────────────────────────────────
# Always the directory that contains *this* file, regardless of CWD.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Data directories ─────────────────────────────────────────────────────────
DATA_DIR         = os.path.join(BASE_DIR, "data")
RAW_DIR          = os.path.join(DATA_DIR, "raw")           # original TN_2023 images
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")  # artifact-removed images
AUGMENTED_DIR    = os.path.join(DATA_DIR, "augmented")      # augmented images
LABELS_DIR       = os.path.join(DATA_DIR, "labels")         # all CSV / XLSX files
CONCEPTS_DIR     = os.path.join(DATA_DIR, "tcav_concepts")  # TCAV concept image sets


# ── Label files ───────────────────────────────────────────────────────────────
ANNOTATIONS_FILE      = os.path.join(LABELS_DIR, "TN_Annotations.xlsx")
LABELS_CSV            = os.path.join(LABELS_DIR, "labels.csv")
AUGMENTED_LABELS_CSV  = os.path.join(LABELS_DIR, "augmented_labels.csv")
AUGMENTED_LABELS_XLSX = os.path.join(LABELS_DIR, "augmented_labels.xlsx")


def biomarker_csv(biomarker: str) -> str:
    """Return the path to the per-biomarker label CSV."""
    return os.path.join(LABELS_DIR, f"{biomarker}.csv")


def concept_dir(biomarker: str) -> str:
    """Return the TCAV concept image directory for *biomarker*."""
    return os.path.join(CONCEPTS_DIR, biomarker.lower())


# ── Models ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")


def model_path(biomarker: str) -> str:
    """Return the path to a trained model checkpoint."""
    return os.path.join(MODELS_DIR, f"{biomarker}_model.pth")


# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def output_dir(biomarker: str) -> str:
    """Return the evaluation output directory for *biomarker*."""
    return os.path.join(OUTPUTS_DIR, biomarker.lower())


# ── Biomarker registry ────────────────────────────────────────────────────────
BIOMARKERS = ["Zoom", "Sagital", "Neutral", "Caliper"]

# Image IDs used for per-image evaluation & GradCAM visualisation
EXAMPLE_IMAGE_IDS = [2, 3, 4, 5, 6, 8, 10, 1, 7, 9, 17, 23, 24, 27]


# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────
SCORE_WEIGHTS = {
    "similarity":  0.40,
    "robustness":  0.10,
    "consistency": 0.15,
    "threshold":   0.15,
    "tcav":        0.20,
}
