"""
Train binary classifiers for each of the 4 biomarkers using the augmented dataset.

Expects:
  - dataset_augmented/ folder with images (from run_preprocessing_pipeline.py)
  - Per-biomarker CSVs: Zoom.csv, Sagital.csv, Neutral.csv, Caliper.csv

Produces:
  - {Biomarker}_model.pth  for each biomarker
  - training_summary.csv   with AUC / sensitivity / specificity
"""

import pandas as pd
from train_model import train_biomarker_classifier

BIOMARKERS = ["Zoom", "Sagital", "Neutral", "Caliper"]
IMG_DIR = "dataset_augmented"

if __name__ == "__main__":
    results = {}

    for biomarker in BIOMARKERS:
        print(f"\n{'=' * 70}")
        print(f"Processing biomarker: {biomarker}")
        print(f"{'=' * 70}")

        result = train_biomarker_classifier(
            biomarker_name=biomarker,
            biomarker_csv=f"{biomarker}.csv",
            img_dir=IMG_DIR,
            batch_size=32,
            num_epochs=50,
            learning_rate=1e-4,
            patience=10,
            output_model_path=f"{biomarker}_model.pth",
        )

        results[biomarker] = result

        print(f"\n{biomarker} training completed!")
        print(f"  AUC:         {result['metrics']['auc']:.4f}")
        print(f"  Sensitivity: {result['metrics']['sensitivity']:.4f}")
        print(f"  Specificity: {result['metrics']['specificity']:.4f}")

    # Save summary
    summary_df = pd.DataFrame(
        {
            "Biomarker": list(results.keys()),
            "AUC": [results[b]["metrics"]["auc"] for b in results],
            "Sensitivity": [results[b]["metrics"]["sensitivity"] for b in results],
            "Specificity": [results[b]["metrics"]["specificity"] for b in results],
            "Model_Path": [results[b]["model_path"] for b in results],
        }
    )

    summary_df.to_csv("training_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("All biomarkers trained successfully!")
    print("=" * 70)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
