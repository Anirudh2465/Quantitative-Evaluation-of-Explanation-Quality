import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from skimage import morphology
import warnings
warnings.filterwarnings('ignore')

def evaluate_cam_comprehensive(biomarker, image, n_perturbations=5, device='cpu', 
                             thresholds=[0.2, 0.4, 0.6, 0.8]):
    """
    Comprehensive CAM quality evaluation combining robustness, consistency, and threshold metrics.
    
    Args:
        biomarker (str): Name of the biomarker (used for model path)
        image (str): Name of the image file (without extension)
        n_perturbations (int): Number of perturbations for robustness testing
        device (str): Device to run on ('cpu' or 'cuda')
        thresholds (list): Activation thresholds to test
    
    Returns:
        tuple: (robustness_score, consistency_score, threshold_score)
    """
    image_path = image
    import os, sys
    # Resolve model path: prefer config.model_path() (new structure),
    # then fall back to legacy root-level names for backward compatibility.
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    try:
        import config as _cfg
        model_path = _cfg.model_path(biomarker)
    except Exception:
        model_path = None

    if model_path is None or not os.path.exists(model_path):
        # Legacy fallbacks (root directory)
        for candidate in [
            f"{biomarker}_model.pth",
            f"{biomarker}.pth",
            f"checkpoint_{biomarker}.pth",
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No model checkpoint found for '{biomarker}'. "
                f"Expected at: {_cfg.model_path(biomarker)}"
            )

    # Setup model
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

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    
    # Define target layer
    target_layer = model.features.denseblock4.denselayer16.conv2
    
    # Initialize GradCAM
    gradcam = GradCAM(model=model, target_layers=[target_layer])
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    
    # Transform for model input (robustness and sharpness)
    transform_basic = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Transform for threshold analysis (normalized)
    transform_normalized = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform_basic(original_image).unsqueeze(0).to(device)
    normalized_tensor = transform_normalized(original_image).unsqueeze(0).to(device)
    
    # Helper function for perturbations
    def apply_perturbations(image_tensor, n_perturbations):
        perturbations = []
        
        for i in range(n_perturbations):
            perturbed = image_tensor.clone()
            perturbation_type = i % 4
            
            if perturbation_type == 0:  # Gaussian noise
                noise = torch.randn_like(perturbed) * 0.01
                perturbed += noise
                
            elif perturbation_type == 1:  # Brightness adjustment
                brightness_factor = 0.9 + np.random.random() * 0.2
                perturbed *= brightness_factor
                
            elif perturbation_type == 2:  # Slight blur
                img_np = perturbed.squeeze().permute(1, 2, 0).cpu().numpy()
                img_np = cv2.GaussianBlur(img_np, (3, 3), 0.5)
                perturbed = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
                
            elif perturbation_type == 3:  # Small rotation simulation
                geometric_noise = torch.randn_like(perturbed) * 0.005
                perturbed += geometric_noise
            
            perturbed = torch.clamp(perturbed, 0, 1)
            perturbations.append(perturbed)
            
        return perturbations
    
    def get_model_prediction(image_tensor):
        """Get model prediction and confidence"""
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            confidence = torch.sigmoid(output).item() if not isinstance(model.classifier[-1], nn.Sigmoid) else output.item()
            prediction = 1 if confidence > 0.5 else 0
        return prediction, confidence
    
    def extract_activation_regions(image_tensor, thresholds):
        """Extract regions with high activations at multiple thresholds"""
        targets = [ClassifierOutputTarget(0)]
        cam = gradcam(input_tensor=image_tensor, targets=targets)[0]
        
        img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        extracted_regions = {}
        
        for threshold in thresholds:
            binary_mask = (cam > threshold).astype(np.uint8)
            cleaned_mask = morphology.remove_small_objects(
                binary_mask.astype(bool), min_size=50
            ).astype(np.uint8)
            
            labeled_mask, num_features = ndimage.label(cleaned_mask)
            
            regions_info = {
                'threshold': threshold,
                'binary_mask': cleaned_mask,
                'labeled_mask': labeled_mask,
                'num_regions': num_features,
                'region_crops': [],
                'region_masks': [],
                'bounding_boxes': []
            }
            
            for region_id in range(1, num_features + 1):
                region_mask = (labeled_mask == region_id).astype(np.uint8)
                coords = np.where(region_mask)
                
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    
                    padding = 10
                    y_min = max(0, y_min - padding)
                    y_max = min(img_np.shape[0], y_max + padding)
                    x_min = max(0, x_min - padding)
                    x_max = min(img_np.shape[1], x_max + padding)
                    
                    crop = img_np[y_min:y_max, x_min:x_max]
                    crop_mask = region_mask[y_min:y_max, x_min:x_max]
                    
                    regions_info['region_crops'].append(crop)
                    regions_info['region_masks'].append(crop_mask)
                    regions_info['bounding_boxes'].append((x_min, y_min, x_max, y_max))
            
            extracted_regions[threshold] = regions_info
        
        return cam, extracted_regions
    
    def create_region_variants(original_image, region_info):
        """Create different variants of extracted regions"""
        variants = []
        
        for i, (crop, mask, bbox) in enumerate(zip(
            region_info['region_crops'], 
            region_info['region_masks'],
            region_info['bounding_boxes']
        )):
            if crop.size == 0:
                continue
            
            # Isolated crop
            isolated_crop = Image.fromarray((crop * 255).astype(np.uint8))
            variants.append({
                'type': 'isolated_crop',
                'image': isolated_crop,
                'region_id': i,
                'description': f'Isolated crop of region {i+1}'
            })
            
            # Masked original
            masked_original = original_image.copy()
            full_mask = np.zeros((224, 224))
            x_min, y_min, x_max, y_max = bbox
            full_mask[y_min:y_max, x_min:x_max] = mask
            
            mask_3d = np.stack([full_mask, full_mask, full_mask], axis=2)
            original_np = np.array(masked_original.resize((224, 224))) / 255.0
            masked_np = original_np * 0.3 + original_np * mask_3d * 0.7
            
            masked_image = Image.fromarray((masked_np * 255).astype(np.uint8))
            variants.append({
                'type': 'masked_original',
                'image': masked_image,
                'region_id': i,
                'description': f'Original with region {i+1} highlighted'
            })
        
        return variants
    
    # 1. Compute Robustness Score
    targets = [ClassifierOutputTarget(0)]
    original_cam = gradcam(input_tensor=input_tensor, targets=targets)[0]
    
    perturbations = apply_perturbations(input_tensor, n_perturbations)
    similarities = []
    
    for perturbed_tensor in perturbations:
        try:
            perturbed_cam = gradcam(input_tensor=perturbed_tensor, targets=targets)[0]
            similarity = ssim(original_cam, perturbed_cam, data_range=1.0)
            similarities.append(max(0, similarity))
        except Exception:
            continue
    
    robustness_score = np.mean(similarities) if similarities else 0.0
    
    # 2. Compute Consistency Score (instead of sharpness)
    # This will be calculated during threshold analysis
    
    # 3. Compute Consistency and Threshold Scores
    # Get original prediction using normalized tensor
    original_pred, original_conf = get_model_prediction(normalized_tensor)
    
    # Extract activation regions using basic tensor
    cam, extracted_regions = extract_activation_regions(input_tensor, thresholds)
    
    # Validate regions and compute both consistency and threshold scores
    threshold_confidences = []
    all_predictions = []
    
    for threshold, region_info in extracted_regions.items():
        if region_info['num_regions'] == 0:
            continue
        
        region_variants = create_region_variants(original_image, region_info)
        region_confidences = []
        region_predictions = []
        
        for variant in region_variants:
            try:
                variant_tensor = transform_normalized(variant['image']).unsqueeze(0)
                pred, conf = get_model_prediction(variant_tensor)
                region_confidences.append(conf)
                region_predictions.append(pred)
                all_predictions.append(pred)
            except Exception:
                continue
        
        if region_confidences:
            avg_confidence = np.mean(region_confidences)
            conf_preservation = 1 - abs(avg_confidence - original_conf)
            threshold_confidences.append(max(0, conf_preservation))
    
    # Consistency Score: Average consistency of predictions across all regions
    consistency_score = np.mean([pred == original_pred for pred in all_predictions]) if all_predictions else 0.0
    
    # Threshold Score: How well confidence is preserved across thresholds
    threshold_score = np.mean(threshold_confidences) if threshold_confidences else 0.0
    
    return robustness_score, consistency_score, threshold_score