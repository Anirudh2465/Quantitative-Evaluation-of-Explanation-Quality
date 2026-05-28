# Quantitative Evaluation of Explanation Quality

Ultrasound biomarker classifier training and explainability evaluation pipeline
for the TN_2023 thyroid nodule dataset.

## Project Structure

```
├── config.py                    ← Central config: all paths, constants, weights
├── main.py                      ← Full pipeline orchestrator
│
├── scripts/                     ← CLI entry-point runners
│   ├── run_preprocessing.py     ← Preprocess + augment TN_2023 images
│   ├── run_training.py          ← Train DenseNet-121 classifiers
│   └── run_evaluation.py        ← Evaluate explanation quality (Grad-CAM + TCAV + scores)
│
├── src/                         ← Core library (importable package)
│   ├── robust_consist_thresh.py ← Robustness, Consistency & Threshold metrics
│   ├── similarity_scoring.py    ← Superpixel Graph Similarity scorer
│   ├── tcav_scoring.py          ← TCAV concept alignment scorer
│   └── train_model.py           ← DenseNet-121 training logic & dataset class
│
├── data/                        ← All data artefacts (git-ignored)
│   ├── raw/                     ← Original TN_2023 images (image_N.jpg)
│   ├── preprocessed/            ← Artifact-removed images (N.jpg)
│   ├── augmented/               ← Augmented images (N.jpg)
│   ├── labels/                  ← All CSV / XLSX label files
│   └── tcav_concepts/           ← TCAV concept images
│       ├── sagittal/present|absent/
│       ├── zoom/present|absent/
│       ├── neutral/present|absent/
│       └── caliper/present|absent/
│
├── models/                      ← Trained .pth checkpoints (git-ignored)
│   ├── Sagital_model.pth
│   ├── Zoom_model.pth
│   ├── Neutral_model.pth
│   ├── Caliper_model.pth
│   └── training_summary.csv
│
└── outputs/                     ← Generated visualisations & CSVs (git-ignored)
    ├── sagittal/
    ├── zoom/
    ├── neutral/
    └── caliper/
```

## Quick Start

### 1 — Run the full pipeline
```bash
python main.py
```

### 2 — Run individual stages
```bash
# Stage 1: Preprocess + augment images
python scripts/run_preprocessing.py

# Stage 2: Train classifiers
python scripts/run_training.py

# Stage 3: Evaluate explanation quality
python scripts/run_evaluation.py --biomarker Sagital   # one biomarker
python scripts/run_evaluation.py                       # all 4 biomarkers
```

## Evaluation Metrics & Weights

| Metric | Weight | Description |
|---|---|---|
| Superpixel Graph Similarity | 40 % | Structural consistency of saliency maps |
| TCAV Concept Expression | 20 % | Continuous CAV-projection score |
| Consistency | 15 % | Stability across perturbations |
| Threshold Score | 15 % | Robustness across activation thresholds |
| Robustness | 10 % | Sensitivity to input noise |

## TCAV Concept Images

Before running evaluation, populate the concept image directories:
- `data/tcav_concepts/<biomarker>/present/` — images **with** the concept
- `data/tcav_concepts/<biomarker>/absent/`  — images **without** the concept
