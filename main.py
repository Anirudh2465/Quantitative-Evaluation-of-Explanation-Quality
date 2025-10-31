import pandas as pd
import numpy as np
import torch

from augment import augment_images
from pre_process import preprocess_images
from train_model import train_biomarker_classifier
from robust_consist_thresh import evaluate_cam_comprehensive
from similarity_scoring import FocusedGradCAMSimilarityScorer


if __name__ == "__main__":
    # Preprocess images
    input_folder = r"\dataset"
    output_folder = r"\dataset_preprocessed"
    preprocess_images(input_folder, output_folder)

    # Augment images
    augment_input_folder = output_folder
    augment_output_folder = r"\dataset_augmented"
    LABEL_FILE = "labels.csv"
    augment_images(augment_input_folder, augment_output_folder, LABEL_FILE, "augmented_labels.csv")
    
    Labels = ['Zoom','Sagital','Neutral','Caliper']
    for i in Labels:
        df = pd.read_csv("augmented_labels.csv")[['Image ID', i]]
        df.to_csv(f"{i}.csv", index=False)
    
    results = {}
    for biomarker in Labels:
        print(f"\n{'='*70}")
        print(f"Processing biomarker: {biomarker}")
        print(f"{'='*70}")
        
        result = train_biomarker_classifier(
            biomarker_name=biomarker,
            biomarker_csv=f"{biomarker}.csv",
            img_dir=augment_output_folder,
            batch_size=32,
            num_epochs=50,
            learning_rate=1e-4,
            patience=10,
            output_model_path=f"{biomarker}_model.pth"
        )
        
        results[biomarker] = result
        
        print(f"\n{biomarker} training completed!")
        print(f"Final AUC: {result['metrics']['auc']:.4f}")
        print(f"Sensitivity: {result['metrics']['sensitivity']:.4f}")
        print(f"Specificity: {result['metrics']['specificity']:.4f}")
    
    # Save summary results
    summary_df = pd.DataFrame({
        'Biomarker': list(results.keys()),
        'AUC': [results[b]['metrics']['auc'] for b in results.keys()],
        'Sensitivity': [results[b]['metrics']['sensitivity'] for b in results.keys()],
        'Specificity': [results[b]['metrics']['specificity'] for b in results.keys()],
        'Model_Path': [results[b]['model_path'] for b in results.keys()]
    })
    
    summary_df.to_csv('training_summary.csv', index=False)
    print("\n" + "="*70)
    print("All biomarkers trained successfully!")
    print("="*70)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    
    example_file = [2, 3, 4, 5, 6, 8, 10, 1, 7, 9, 17, 23, 24, 27]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in Labels:
        for j in example_file:
            biomarker_name = i
            biomarker_path = f"{i}.jpg"
            model_path = f"{i}.pth"
            file_name = f"dataset_preprocessed/{j}.jpg"
            print(f"Evaluating {biomarker_name} on image {file_name}")
            robustness, consistency, threshold = evaluate_cam_comprehensive(biomarker=biomarker_name,image=file_name,n_perturbations=5,device=device,thresholds=[0.2, 0.4, 0.6, 0.8])
            scorer = FocusedGradCAMSimilarityScorer(reference_image_path=biomarker_name, model_path=model_path)
            similarity_score = scorer.visualize_focused_comparison(file_name, None)
            
            Final_Score = (0.5 * similarity_score) + (robustness*0.1) + (consistency*0.2) + (threshold*0.2)
            print("Final Score is :", Final_Score)