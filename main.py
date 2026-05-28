"""
Full pipeline orchestrator.

Runs the three stages in sequence:
  1. Preprocess raw images (TN_2023 → data/preprocessed → data/augmented)
  2. Train binary classifiers for all 4 biomarkers
  3. Evaluate explanation quality for all 4 biomarkers

Individual stages can be run independently:
    python scripts/run_preprocessing.py
    python scripts/run_training.py
    python scripts/run_evaluation.py --biomarker Sagital
    python scripts/run_evaluation.py              # all biomarkers
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from scripts.run_preprocessing import main as preprocess
from scripts.run_training      import run_training
from scripts.run_evaluation    import evaluate_biomarker


def main() -> None:
    print("=" * 70)
    print("  Full Explanation Quality Pipeline")
    print("=" * 70)

    # ── Stage 1: preprocess ───────────────────────────────────────────────
    print("\n>>> STAGE 1: Preprocessing & Augmentation")
    preprocess()

    # ── Stage 2: train ────────────────────────────────────────────────────
    print("\n>>> STAGE 2: Training Classifiers")
    run_training()

    # ── Stage 3: evaluate ─────────────────────────────────────────────────
    print("\n>>> STAGE 3: Evaluating Explanation Quality")
    for biomarker in config.BIOMARKERS:
        evaluate_biomarker(biomarker)

    print("\n" + "=" * 70)
    print("  FULL PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()