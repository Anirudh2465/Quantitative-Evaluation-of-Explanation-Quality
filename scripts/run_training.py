"""
Train binary classifiers for all 4 biomarkers using the augmented dataset.

Expects:
  - data/augmented/   — images produced by scripts/run_preprocessing.py
  - data/labels/{Biomarker}.csv — per-biomarker label files

Produces:
  - models/{Biomarker}_model.pth      for each biomarker
  - models/training_summary.csv       with AUC / sensitivity / specificity

Run standalone:
    python scripts/run_training.py
Or call run_training() from main.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import config
from src.train_model import train_biomarker_classifier


def run_training() -> dict:
    """Train all biomarkers and return a dict of results."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    results = {}

    for biomarker in config.BIOMARKERS:
        print(f"\n{'=' * 70}")
        print(f"Training: {biomarker}")
        print(f"{'=' * 70}")

        result = train_biomarker_classifier(
            biomarker_name=biomarker,
            biomarker_csv=config.biomarker_csv(biomarker),
            img_dir=config.AUGMENTED_DIR,
            batch_size=32,
            num_epochs=50,
            learning_rate=1e-4,
            patience=10,
            output_model_path=config.model_path(biomarker),
        )

        results[biomarker] = result
        m = result["metrics"]
        print(f"  AUC:         {m['auc']:.4f}")
        print(f"  Sensitivity: {m['sensitivity']:.4f}")
        print(f"  Specificity: {m['specificity']:.4f}")

    # Persist training summary
    summary_df = pd.DataFrame({
        "Biomarker":   list(results.keys()),
        "AUC":         [results[b]["metrics"]["auc"]         for b in results],
        "Sensitivity": [results[b]["metrics"]["sensitivity"] for b in results],
        "Specificity": [results[b]["metrics"]["specificity"] for b in results],
        "Model_Path":  [results[b]["model_path"]             for b in results],
    })
    summary_path = os.path.join(config.MODELS_DIR, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("All biomarkers trained successfully!")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    return results


if __name__ == "__main__":
    run_training()
