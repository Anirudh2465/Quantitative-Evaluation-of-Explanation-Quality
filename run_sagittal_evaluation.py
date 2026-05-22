import os
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
from tcav_scoring import TCAVScorer

def main():
    print("="*60)
    print("Starting Sagittal Model TCAV Evaluation")
    print("="*60)

    # Configuration
    model_path = "Sagital_model.pth"
    biomarker_name = "Sagital"
    concept_csv = "Sagital.csv"
    img_dir_augmented = "dataset_augmented"
    img_dir_preprocessed = "dataset_preprocessed"
    output_dir = "TCAV_outputs"
    concept_dir = "tcav_train/sagittal"
    example_files = [2, 3, 4, 5, 6, 8, 10, 1, 7, 9, 17, 23, 24, 27]
    device = torch.device('cpu')
    n_random_runs = 10
    
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    print(f"Concept CSV: {concept_csv}")
    print(f"Augmented image dir: {img_dir_augmented}")
    print(f"Preprocessed image dir: {img_dir_preprocessed}")
    print(f"Custom Concept Dir: {concept_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Example files: {example_files}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check model path
    if not os.path.exists(model_path):
        # try lowercase / uppercase variations
        if os.path.exists("Sagittal_model.pth"):
            model_path = "Sagittal_model.pth"
        elif os.path.exists("checkpoint_Sagital.pth"):
            model_path = "checkpoint_Sagital.pth"
        else:
            print(f"Error: Model path '{model_path}' not found!")
            return

    # Load model for prediction and Grad-CAM
    print("\nLoading DenseNet-121 model...")
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Target layer for Grad-CAM
    target_layer = model.features.denseblock4.denselayer16.conv2
    gradcam = GradCAM(model=model, target_layers=[target_layer])
    
    # Define transforms
    transform_basic = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    transform_normalized = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load TCAV Scorer (instantiate once)
    print("\nInitializing TCAV Scorer...")
    tcav_scorer = TCAVScorer(
        model_path=model_path,
        biomarker_name=biomarker_name,
        concept_csv=concept_csv,
        img_dir=img_dir_augmented,
        device=device,
        concept_dir=concept_dir
    )
    
    # Get all IDs to form small test sets
    df_concept = pd.read_csv(concept_csv)
    df_concept.columns = [c.strip() for c in df_concept.columns]
    all_concept_ids = df_concept["Image ID"].tolist()
    
    results = []
    
    for idx, img_id in enumerate(example_files):
        print(f"\nProcessing Image {img_id} ({idx+1}/{len(example_files)})...")
        img_filename = f"{img_id}.jpg"
        img_path = os.path.join(img_dir_preprocessed, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_path} not found. Skipping.")
            continue
            
        # 1. Generate Prediction and Grad-CAM
        original_image = Image.open(img_path).convert('RGB')
        input_tensor = transform_basic(original_image).unsqueeze(0).to(device)
        normalized_tensor = transform_normalized(original_image).unsqueeze(0).to(device)
        
        # Get prediction and confidence
        with torch.no_grad():
            output = model(normalized_tensor)
            confidence = output.item()
            pred = 1 if confidence > 0.5 else 0
            
        # Retrieve true label and prediction strings
        true_val = df_concept.loc[df_concept["Image ID"] == img_id, biomarker_name].values[0]
        true_label_str = "Present" if true_val == 1 else "Absent"
        pred_label_str = "Present" if pred == 1 else "Absent"
            
        # Get Grad-CAM
        targets = [ClassifierOutputTarget(0)]
        cam = gradcam(input_tensor=input_tensor, targets=targets)[0]
        
        # Build test set: this image + neighbors of the SAME class
        test_ids = [img_id]
        same_class_ids = df_concept.loc[df_concept[biomarker_name] == true_val, "Image ID"].tolist()
        for cid in same_class_ids:
            if cid != img_id:
                test_ids.append(cid)
            if len(test_ids) >= 15:
                break
                
        mean_tcav_score, p_value, is_significant, details = \
            tcav_scorer.compute_tcav_with_statistical_testing(
                test_image_ids=test_ids,
                n_random_runs=n_random_runs
            )
            
        # Score dampening if not statistically significant
        dampened_score = mean_tcav_score
        if not is_significant:
            dampened_score = 0.5 + (mean_tcav_score - 0.5) * 0.5
            
        mean_cav_acc = details["mean_cav_accuracy"]
        
        print(f"  Prediction: {pred} (Conf: {confidence:.4f})")
        print(f"  TCAV Score (raw): {mean_tcav_score:.4f} (dampened: {dampened_score:.4f})")
        print(f"  CAV Accuracy: {mean_cav_acc:.4f}, p-value: {p_value:.4f}, Significant: {is_significant}")
        
        # Save records
        results.append({
            "Image ID": img_id,
            "Prediction": pred,
            "Confidence": confidence,
            "TCAV Score (Raw)": mean_tcav_score,
            "TCAV Score (Dampened)": dampened_score,
            "CAV Accuracy": mean_cav_acc,
            "p-value": p_value,
            "Significant": is_significant,
            "Original Label": true_label_str
        })
        
        # 3. Create side-by-side visualization
        original_np = np.array(original_image.resize((224, 224))) / 255.0
        
        # Generate color heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.6 * original_np + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='#111111')
        fig.suptitle(
            f"Image {img_id} — Sagittal TCAV Evaluation", 
            color='white', fontsize=16, fontweight='bold', y=0.98
        )
        
        # Panel 1: Original
        axes[0].imshow(original_np)
        axes[0].set_title("Original Image", color='#cccccc', fontsize=12)
        axes[0].axis('off')
        
        # Panel 2: Grad-CAM
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM Overlay", color='#cccccc', fontsize=12)
        axes[1].axis('off')
        
        # Info Box (Add text at the bottom/side)
        info_text = (
            f"TCAV Score: {dampened_score:.4f}\n"
            f"Prediction: {pred_label_str} (Conf: {confidence:.2%})  |  Original Label: {true_label_str}"
        )
        
        fig.text(
            0.5, 0.02, info_text, 
            color='white', ha='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#222222', edgecolor='#444444')
        )
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
        
        save_path = os.path.join(output_dir, f"{img_id}_tcav.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#111111')
        plt.close()
        
        print(f"  Saved plot to {save_path}")
        
    # 4. Save CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "tcav_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV results to {csv_path}")
    
    # 5. Generate Markdown Report
    print("Generating Markdown report...")
    report_path = os.path.join(output_dir, "README.md")
    
    # Compute averages
    avg_raw_tcav = results_df["TCAV Score (Raw)"].mean()
    avg_damp_tcav = results_df["TCAV Score (Dampened)"].mean()
    avg_cav_acc = results_df["CAV Accuracy"].mean()
    sig_pct = results_df["Significant"].mean() * 100
    
    markdown_content = f"""# Sagittal Model TCAV Evaluation Report

This report presents concept alignment analysis for the Sagittal biomarker classifier model (`Sagital_model.pth`) using **Testing with Concept Activation Vectors (TCAV)**.

The concept was trained using manually cropped slices from:
- Concept present: `{os.path.join(concept_dir, 'present')}`
- Concept absent: `{os.path.join(concept_dir, 'absent')}`

## Performance & Statistical Summary

- **Average TCAV Score (Raw)**: {avg_raw_tcav:.4f}
- **Average TCAV Score (Dampened)**: {avg_damp_tcav:.4f} (Adjusted for significance)
- **Average CAV Classifier Accuracy**: {avg_cav_acc:.2%} (Learnability of concept representation)
- **Statistical Significance Rate**: {sig_pct:.1f}% of test runs are statistically significant ($p < 0.05$)

---

## Detailed Evaluation Table

| Image ID | Prediction | Confidence | Original Label | TCAV Score (Raw) | TCAV Score (Dampened) | CAV Accuracy | p-value | Significant |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    
    for r in results:
        pred_label = "**Sagital**" if r["Prediction"] == 1 else "Non-Sagital"
        sig_label = "✅ Yes" if r["Significant"] else "❌ No"
        markdown_content += (
            f"| {r['Image ID']} | {pred_label} | {r['Confidence']:.2%} | {r['Original Label']} | "
            f"{r['TCAV Score (Raw)']:.4f} | {r['TCAV Score (Dampened)']:.4f} | "
            f"{r['CAV Accuracy']:.2%} | {r['p-value']:.4f} | {sig_label} |\n"
        )
        
    markdown_content += """
---

## Visualizations

Below are the side-by-side original image and Grad-CAM overlay comparisons for each evaluated image, along with their individual TCAV metrics.

"""
    
    # Add screenshots links
    for r in results:
        img_id = r["Image ID"]
        pred_label_str = "Present" if r["Prediction"] == 1 else "Absent"
        markdown_content += f"""### Image {img_id}
TCAV Score: **{r['TCAV Score (Dampened)']:.4f}** | Prediction: **{pred_label_str}** | Original Label: **{r['Original Label']}**

![Image {img_id} TCAV Plot]({img_id}_tcav.png)

"""
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"Saved Markdown report to {report_path}")
    print("\nTCAV Evaluation completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
