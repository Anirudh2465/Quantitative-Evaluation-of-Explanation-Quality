"""
Unified explanation quality evaluation for any (or all) biomarkers.

Usage:
    python scripts/run_evaluation.py                     # evaluate all 4 biomarkers
    python scripts/run_evaluation.py --biomarker Sagital
    python scripts/run_evaluation.py --biomarker Zoom
    python scripts/run_evaluation.py --biomarker Neutral
    python scripts/run_evaluation.py --biomarker Caliper

Outputs per biomarker (written to outputs/<biomarker>/):
  - {id}_gradcam.png        original image + Grad-CAM overlay, annotated with scores
  - {id}_superpixel.png     superpixel graph similarity visualisation
  - {biomarker}_evaluation_results.csv
"""

import sys
import os
import argparse

# Make the project root importable when this script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from src.robust_consist_thresh import evaluate_cam_comprehensive
from src.similarity_scoring import SuperpixelGraphSimilarityScorer
from src.tcav_scoring import evaluate_tcav


# ── Model ─────────────────────────────────────────────────────────────────────

def _build_model(checkpoint: str, device: torch.device) -> nn.Module:
    """Return a DenseNet-121 with the project classifier head, ready for inference."""
    net = models.densenet121(pretrained=False)
    net.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(net.classifier.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    return net.eval().to(device)


# ── Activation extraction ─────────────────────────────────────────────────────

def _extract_bottleneck_activation(
    model: nn.Module,
    target_layer: nn.Module,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
) -> np.ndarray:
    """Hook *target_layer*, run a forward pass, return a GAP-pooled 1-D vector."""
    captured: dict = {}

    def _hook(module, inp, out):
        captured["act"] = out

    handle = target_layer.register_forward_hook(_hook)
    try:
        img    = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            model(tensor)
        act = torch.nn.functional.adaptive_avg_pool2d(captured["act"], 1)
        return act.view(-1).cpu().numpy()
    finally:
        handle.remove()


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate_biomarker(biomarker: str) -> None:
    """Run the full two-pass evaluation pipeline for *biomarker*."""
    print("\n" + "=" * 70)
    print(f"  {biomarker} -- Explanation Quality Evaluation")
    print("=" * 70)

    checkpoint  = config.model_path(biomarker)
    csv_path    = config.biomarker_csv(biomarker)
    concept_dir = config.concept_dir(biomarker)
    out_dir     = config.output_dir(biomarker)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cpu")
    print(f"Device : {device}")
    print(f"Model  : {checkpoint}")

    # ── Initialise model & tools ──────────────────────────────────────────
    model        = _build_model(checkpoint, device)
    target_layer = model.features.denseblock4.denselayer16.conv2
    gradcam      = GradCAM(model=model, target_layers=[target_layer])

    transform_basic = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    transform_norm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    df_labels = pd.read_csv(csv_path)
    df_labels.columns = [c.strip() for c in df_labels.columns]

    reference_path = os.path.join(config.PREPROCESSED_DIR,
                                  f"{config.EXAMPLE_IMAGE_IDS[0]}.jpg")
    print("\nInitialising Superpixel Graph Similarity Scorer ...")
    similarity_scorer = SuperpixelGraphSimilarityScorer(
        model_path=checkpoint,
        reference_image_path=reference_path,
        n_superpixels=200,
        compactness=20,
    )

    # ── Pass 1: base metrics & raw CAV projections ────────────────────────
    print("\n" + "-" * 50)
    print("Pass 1 -- Base Metrics & Activation Projections")
    print("-" * 50)

    raw_data:    list = []
    projections: list = []

    for idx, img_id in enumerate(config.EXAMPLE_IMAGE_IDS):
        print(f"\nImage {img_id}  ({idx + 1}/{len(config.EXAMPLE_IMAGE_IDS)})")
        file_name = os.path.join(config.PREPROCESSED_DIR, f"{img_id}.jpg")

        if not os.path.exists(file_name):
            print(f"  WARNING: {file_name} not found - skipping.")
            continue

        row        = df_labels[df_labels["Image ID"] == img_id]
        orig_label = int(row[biomarker].values[0]) if len(row) > 0 else -1

        original_image = Image.open(file_name).convert("RGB")
        input_tensor   = transform_basic(original_image).unsqueeze(0).to(device)
        norm_tensor    = transform_norm(original_image).unsqueeze(0).to(device)

        with torch.no_grad():
            probability = model(norm_tensor).item()
        pred_label = 1 if probability > 0.5 else 0

        # Robustness / Consistency / Threshold
        print("  [1] Robustness / Consistency / Threshold ...")
        robustness, consistency, threshold = evaluate_cam_comprehensive(
            biomarker=biomarker, image=file_name,
            n_perturbations=5, device=device,
            thresholds=[0.2, 0.4, 0.6, 0.8],
        )

        # Superpixel graph similarity
        print("  [2] Superpixel Graph Similarity ...")
        sim_save = os.path.join(out_dir, f"{img_id}_superpixel.png")
        similarity_score = similarity_scorer.visualize_focused_comparison(file_name, sim_save)

        # TCAV
        print("  [3] TCAV Concept Alignment ...")
        tcav_result = evaluate_tcav(
            biomarker=biomarker, image=file_name,
            img_dir=config.AUGMENTED_DIR,
            biomarker_csv=csv_path,
            device=device,
            concept_dir=concept_dir,
            n_random_runs=50,
            cav_accuracy_threshold=0.70,
        )

        # Raw CAV projection
        best_cav = tcav_result.details.get("cav")
        if best_cav is not None and best_cav.size > 1:
            act        = _extract_bottleneck_activation(
                model, target_layer, file_name, transform_norm, device)
            projection = float(np.dot(act, best_cav))
        else:
            print("  WARNING: CAV unavailable - projection set to 0.0")
            projection = 0.0

        print(f"  Raw projection: {projection:.4f}")
        projections.append(projection)

        raw_data.append(dict(
            img_id=img_id,
            orig_label=orig_label, pred_label=pred_label, probability=probability,
            similarity_score=similarity_score,
            robustness=robustness, consistency=consistency, threshold=threshold,
            raw_projection=projection,
            original_image=original_image, input_tensor=input_tensor,
        ))

    if not projections:
        print("ERROR: No images were processed. Check data paths.")
        return

    # ── Pass 2: normalise & visualise ─────────────────────────────────────
    print("\n" + "-" * 50)
    print("Pass 2 -- Normalising & Generating Visualisations")
    print("-" * 50)

    proj_min, proj_max = min(projections), max(projections)
    print(f"Projection range: [{proj_min:.4f}, {proj_max:.4f}]")

    w       = config.SCORE_WEIGHTS
    results = []

    for data in raw_data:
        img_id   = data["img_id"]
        raw_proj = data["raw_projection"]
        tcav_s   = (
            (raw_proj - proj_min) / (proj_max - proj_min)
            if proj_max > proj_min else 0.5
        )

        sim  = data["similarity_score"]
        rob  = data["robustness"]
        cons = data["consistency"]
        thr  = data["threshold"]

        final = (
            w["similarity"]  * sim  +
            w["robustness"]  * rob  +
            w["consistency"] * cons +
            w["threshold"]   * thr  +
            w["tcav"]        * tcav_s
        )

        print(f"Image {img_id:2d} | tcav={tcav_s:.4f} | final={final:.4f}")

        results.append({
            "Image ID":          img_id,
            "Original Label":    data["orig_label"],
            "Predicted Label":   data["pred_label"],
            "Probability":       data["probability"],
            "Similarity Score":  sim,
            "Robustness Score":  rob,
            "Consistency Score": cons,
            "Threshold Score":   thr,
            "TCAV Score":        tcav_s,
            "Raw Projection":    raw_proj,
            "Final Score":       final,
        })

        # ── Grad-CAM visualisation ─────────────────────────────────────
        cam     = gradcam(input_tensor=data["input_tensor"],
                          targets=[ClassifierOutputTarget(0)])[0]
        orig_np = np.array(data["original_image"].resize((224, 224))) / 255.0
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.6 * orig_np + 0.4 * heatmap, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#111111")
        fig.suptitle(
            f"Image {img_id} - {biomarker} Biomarker Evaluation\n"
            f"Explainability Score: {final:.4f}",
            color="white", fontsize=14, fontweight="bold", y=0.98,
        )
        axes[0].imshow(orig_np);  axes[0].set_title("Original Image",   color="#cccccc", fontsize=12); axes[0].axis("off")
        axes[1].imshow(overlay);  axes[1].set_title("Grad-CAM Overlay", color="#cccccc", fontsize=12); axes[1].axis("off")

        orig_lbl = biomarker if data["orig_label"] == 1 else f"Non-{biomarker}"
        pred_lbl = biomarker if data["pred_label"] == 1 else f"Non-{biomarker}"
        info = (
            f"Original: {orig_lbl}  |  Predicted: {pred_lbl} "
            f"(Conf: {data['probability']:.2%})\n"
            f"sim={sim:.4f}  rob={rob:.4f}  cons={cons:.4f}  "
            f"thr={thr:.4f}  tcav={tcav_s:.4f}"
        )
        fig.text(0.5, 0.02, info, color="white", ha="center", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5",
                           facecolor="#222222", edgecolor="#444444"))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)

        viz_path = os.path.join(out_dir, f"{img_id}_gradcam.png")
        plt.savefig(viz_path, dpi=200, bbox_inches="tight", facecolor="#111111")
        plt.close()
        print(f"  Saved -> {viz_path}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_out = os.path.join(out_dir, f"{biomarker.lower()}_evaluation_results.csv")
    pd.DataFrame(results).to_csv(csv_out, index=False)
    print(f"\nCSV -> {csv_out}")
    print(f"\n{'=' * 70}\n  {biomarker} EVALUATION COMPLETE\n{'=' * 70}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate explanation quality for one or all biomarkers."
    )
    parser.add_argument(
        "--biomarker",
        choices=config.BIOMARKERS + ["all"],
        default="all",
        help="Biomarker to evaluate (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    targets = config.BIOMARKERS if args.biomarker == "all" else [args.biomarker]
    for b in targets:
        evaluate_biomarker(b)
