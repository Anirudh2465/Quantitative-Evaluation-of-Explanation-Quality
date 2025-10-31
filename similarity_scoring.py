import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from skimage import morphology
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.stats import pearsonr
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from skimage import measure
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os


class FocusedGradCAMSimilarityScorer:
    """
    Tool to compute similarity scores focusing only on high-activation GradCAM regions
    and their corresponding image content
    """
    
    def __init__(self, model_path, reference_image_path, activation_threshold=0.6):
        self.model_path = model_path
        self.reference_image_path = reference_image_path
        self.activation_threshold = activation_threshold  # Only consider regions above this threshold
        
        # Load model for new image processing
        self.model = self._load_model()
        if self.model is None:
            raise ValueError("Failed to load model")
        
        # Get target layer
        self.target_layer = self.model.features.denseblock4.denselayer16.conv2
        
        # Initialize GradCAM for new images
        self.gradcam = GradCAM(model=self.model, target_layers=[self.target_layer])
        
        # Preprocessing transforms
        self.viz_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Extract reference focused regions
        self.reference_features = self._extract_reference_focused_features()
        
    def _load_model(self):
        """Load the trained model"""
        try:
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
            
            state_dict = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def _extract_activation_from_colors(self, image):
        """Extract activation values from GradCAM overlay colors with better precision"""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] if needed
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image.astype(np.float32) / 255.0
        
        # Use HSV color space for better color-to-activation mapping
        hsv_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hue = hsv_image[:, :, 0].astype(np.float32)
        saturation = hsv_image[:, :, 1].astype(np.float32) / 255.0
        value = hsv_image[:, :, 2].astype(np.float32) / 255.0
        
        # More precise activation mapping based on GradCAM color scheme
        activation_map = np.zeros_like(hue)
        
        # Red regions (high activation) - hue 0-20 and 340-360 (mapped to 170-180)
        red_mask1 = (hue <= 20) & (saturation > 0.4) & (value > 0.3)
        red_mask2 = (hue >= 170) & (saturation > 0.4) & (value > 0.3)
        red_mask = red_mask1 | red_mask2
        activation_map[red_mask] = 0.8 + 0.2 * value[red_mask]  # 0.8-1.0
        
        # Orange/Yellow regions (medium-high activation) - hue 20-60
        orange_mask = (hue > 20) & (hue <= 60) & (saturation > 0.4) & (value > 0.3)
        activation_map[orange_mask] = 0.5 + 0.3 * value[orange_mask]  # 0.5-0.8
        
        # Green regions (medium activation) - hue 60-120
        green_mask = (hue > 60) & (hue <= 120) & (saturation > 0.4) & (value > 0.3)
        activation_map[green_mask] = 0.2 + 0.3 * value[green_mask]  # 0.2-0.5
        
        # Blue regions (low activation) - hue 120-170
        blue_mask = (hue > 120) & (hue < 170) & (saturation > 0.4) & (value > 0.3)
        activation_map[blue_mask] = 0.0 + 0.2 * value[blue_mask]  # 0.0-0.2
        
        # Low saturation regions (background/neutral)
        gray_mask = (saturation <= 0.4) | (value <= 0.3)
        activation_map[gray_mask] = 0.1  # Low baseline activation
        
        return activation_map, rgb_image
    
    def _generate_gradcam_for_new_image(self, image_path):
        """Generate GradCAM for a new image"""
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.viz_transform(original_image).unsqueeze(0)
        
        # Generate GradCAM
        targets = [ClassifierOutputTarget(0)]
        cam = self.gradcam(input_tensor=input_tensor, targets=targets)[0]
        
        # Convert image tensor to numpy
        img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        
        return img_np, cam, original_image
    
    def _extract_focused_regions(self, image, activation_map):
        """Extract only the most important regions based on activation threshold"""
        target_size = (224, 224)
        
        # Resize to target size
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size)
        if activation_map.shape != target_size:
            activation_map = cv2.resize(activation_map, target_size)
        
        # Convert image to proper format
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        if len(image_uint8.shape) == 2:
            image_uint8 = np.stack([image_uint8] * 3, axis=-1)
        
        # Find high-activation regions
        high_activation_mask = activation_map > self.activation_threshold
        
        if not np.any(high_activation_mask):
            print(f"Warning: No regions found above threshold {self.activation_threshold}")
            # Lower threshold temporarily
            high_activation_mask = activation_map > (self.activation_threshold * 0.7)
        
        # Use connected components to find distinct regions
        labeled_regions = measure.label(high_activation_mask)
        region_props = measure.regionprops(labeled_regions)
        
        focused_regions = []
        
        for i, prop in enumerate(region_props):
            if prop.area < 50:  # Skip very small regions
                continue
            
            # Get region mask
            region_mask = labeled_regions == prop.label
            
            # Extract region features
            region_coords = prop.coords
            min_row, min_col = region_coords.min(axis=0)
            max_row, max_col = region_coords.max(axis=0)
            
            # Region activation statistics
            region_activations = activation_map[region_mask]
            mean_activation = np.mean(region_activations)
            max_activation = np.max(region_activations)
            activation_std = np.std(region_activations)
            
            # Region image content features
            region_image = image_uint8[region_mask]
            
            # Color statistics
            mean_rgb = np.mean(region_image, axis=0)
            std_rgb = np.std(region_image, axis=0)
            
            # Texture features using Local Binary Pattern
            region_gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
            region_gray_masked = region_gray[region_mask]
            
            # Histogram features for the region
            hist_r = np.histogram(region_image[:, 0], bins=16, range=(0, 256))[0].astype(np.float32)
            hist_g = np.histogram(region_image[:, 1], bins=16, range=(0, 256))[0].astype(np.float32)
            hist_b = np.histogram(region_image[:, 2], bins=16, range=(0, 256))[0].astype(np.float32)
            
            # Normalize histograms
            hist_r = hist_r / np.sum(hist_r) if np.sum(hist_r) > 0 else hist_r
            hist_g = hist_g / np.sum(hist_g) if np.sum(hist_g) > 0 else hist_g
            hist_b = hist_b / np.sum(hist_b) if np.sum(hist_b) > 0 else hist_b
            
            # Ensure histograms are float32 after normalization
            hist_r = hist_r.astype(np.float32)
            hist_g = hist_g.astype(np.float32)
            hist_b = hist_b.astype(np.float32)
            
            # Spatial features
            centroid = prop.centroid
            area_ratio = prop.area / (224 * 224)
            aspect_ratio = (max_row - min_row) / max(1, max_col - min_col)
            
            focused_regions.append({
                'centroid': np.array(centroid) / 224.0,  # Normalized
                'area_ratio': area_ratio,
                'aspect_ratio': aspect_ratio,
                'mean_activation': mean_activation,
                'max_activation': max_activation,
                'activation_std': activation_std,
                'mean_rgb': mean_rgb,
                'std_rgb': std_rgb,
                'hist_r': hist_r,
                'hist_g': hist_g,
                'hist_b': hist_b,
                'bbox': (min_row/224.0, min_col/224.0, max_row/224.0, max_col/224.0),
                'mask': region_mask
            })
        
        return focused_regions
    
    def _extract_reference_focused_features(self):
        """Extract focused features from reference overlay image"""
        print(f"Extracting focused features from reference: {self.reference_image_path}")
        
        overlay_image = cv2.imread(self.reference_image_path)
        if overlay_image is None:
            raise ValueError(f"Could not load reference image: {self.reference_image_path}")
        
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        activation_map, rgb_image = self._extract_activation_from_colors(overlay_image)
        
        focused_regions = self._extract_focused_regions(rgb_image, activation_map)
        print(f"Found {len(focused_regions)} focused regions in reference")
        
        return {
            'regions': focused_regions,
            'activation_map': activation_map,
            'image': rgb_image
        }
    
    def _compute_region_similarity(self, ref_region, test_region):
        """Compute similarity between two focused regions"""
        similarities = []
        
        # Activation similarity (most important)
        activation_sim = 1.0 - abs(ref_region['mean_activation'] - test_region['mean_activation'])
        max_activation_sim = 1.0 - abs(ref_region['max_activation'] - test_region['max_activation'])
        activation_weight = 0.4
        similarities.append((activation_sim + max_activation_sim) / 2 * activation_weight)
        
        # Color histogram similarity (very important for image content)
        try:
            hist_sim_r = cv2.compareHist(ref_region['hist_r'].astype(np.float32), 
                                       test_region['hist_r'].astype(np.float32), 
                                       cv2.HISTCMP_CORREL)
            hist_sim_g = cv2.compareHist(ref_region['hist_g'].astype(np.float32), 
                                       test_region['hist_g'].astype(np.float32), 
                                       cv2.HISTCMP_CORREL)
            hist_sim_b = cv2.compareHist(ref_region['hist_b'].astype(np.float32), 
                                       test_region['hist_b'].astype(np.float32), 
                                       cv2.HISTCMP_CORREL)
            
            # Handle NaN values in histogram comparison
            hist_sim_r = hist_sim_r if not np.isnan(hist_sim_r) else 0
            hist_sim_g = hist_sim_g if not np.isnan(hist_sim_g) else 0
            hist_sim_b = hist_sim_b if not np.isnan(hist_sim_b) else 0
            
        except Exception as e:
            print(f"Warning: Histogram comparison failed: {e}")
            hist_sim_r = hist_sim_g = hist_sim_b = 0
        
        color_content_sim = (hist_sim_r + hist_sim_g + hist_sim_b) / 3
        color_content_weight = 0.35
        similarities.append(color_content_sim * color_content_weight)
        
        # Spatial similarity (position and size)
        centroid_dist = np.linalg.norm(ref_region['centroid'] - test_region['centroid'])
        spatial_sim = np.exp(-centroid_dist * 5)  # Exponential decay
        
        area_sim = 1.0 - abs(ref_region['area_ratio'] - test_region['area_ratio']) / (ref_region['area_ratio'] + test_region['area_ratio'] + 1e-6)
        spatial_weight = 0.15
        similarities.append((spatial_sim + area_sim) / 2 * spatial_weight)
        
        # Mean color similarity
        rgb_sim = 1.0 - np.linalg.norm(ref_region['mean_rgb'] - test_region['mean_rgb']) / (255.0 * np.sqrt(3))
        color_weight = 0.1
        similarities.append(rgb_sim * color_weight)
        
        return np.sum(similarities)
    
    def _match_regions(self, ref_regions, test_regions):
        """Match reference regions to test regions using Hungarian algorithm"""
        if len(ref_regions) == 0 or len(test_regions) == 0:
            return [], 0.0
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(ref_regions), len(test_regions)))
        
        for i, ref_region in enumerate(ref_regions):
            for j, test_region in enumerate(test_regions):
                similarity_matrix[i, j] = self._compute_region_similarity(ref_region, test_region)
        
        # Convert similarity to cost (Hungarian algorithm minimizes)
        cost_matrix = 1.0 - similarity_matrix
        
        # Apply Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        ref_indices, test_indices = linear_sum_assignment(cost_matrix)
        
        # Calculate matched similarities
        matched_similarities = []
        matches = []
        
        for ref_idx, test_idx in zip(ref_indices, test_indices):
            sim = similarity_matrix[ref_idx, test_idx]
            matched_similarities.append(sim)
            matches.append((ref_idx, test_idx, sim))
        
        # Penalize unmatched regions
        unmatched_penalty = 0.3
        num_unmatched = abs(len(ref_regions) - len(test_regions))
        
        if len(matched_similarities) > 0:
            avg_similarity = np.mean(matched_similarities)
            # Apply penalty for unmatched regions
            final_similarity = avg_similarity * (1.0 - unmatched_penalty * num_unmatched / max(len(ref_regions), len(test_regions)))
        else:
            final_similarity = 0.0
        
        return matches, final_similarity
    
    def compute_focused_similarity(self, new_image_path):
        """Compute focused similarity score"""
        # Generate GradCAM for new image
        img_np, cam, original_image = self._generate_gradcam_for_new_image(new_image_path)
        
        # Extract focused regions from test image
        test_regions = self._extract_focused_regions(img_np, cam)
        
        if len(test_regions) == 0:
            print("Warning: No focused regions found in test image")
            return 0.0
        
        print(f"Found {len(test_regions)} focused regions in test image")
        
        # Match regions and compute similarity
        matches, similarity_score = self._match_regions(
            self.reference_features['regions'], 
            test_regions
        )
        
        return similarity_score, {
            'test_regions': test_regions,
            'matches': matches,
            'test_cam': cam,
            'test_image': img_np
        }
    
    def visualize_focused_comparison(self, new_image_path, save_path=None):
        """Visualize the focused region comparison - simplified to 2 plots only"""
        similarity_score, details = self.compute_focused_similarity(new_image_path)
        image_name = os.path.splitext(os.path.basename(new_image_path))[0]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Test image
        test_img = (details['test_image'] * 255).astype(np.uint8) if details['test_image'].max() <= 1 else details['test_image']
        
        axes[0].imshow(test_img)
        axes[0].set_title(f'Image {image_name}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Highlight focused regions in test image
        test_focused = test_img.copy().astype(np.float32)
        if test_focused.max() > 1.0:
            test_focused = test_focused / 255.0
            
        for region in details['test_regions']:
            mask = region['mask']
            if mask.shape == test_focused.shape[:2]:  # Ensure mask matches image dimensions
                test_focused[mask] = test_focused[mask] * 0.7 + np.array([0, 1.0, 0]) * 0.3
        
        axes[1].imshow(np.clip(test_focused, 0, 1))
        axes[1].set_title(f'Focused Regions', 
                        fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Set default save path if not provided
        if save_path is None:
            save_dir = r"D:\Internship\Jithin sir Sikha ma'am\FINAL ONLY. NO MORE CHANGES\gradcam_outputs\{biomarker}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Extract filename from new_image_path for unique naming
            image_name = os.path.splitext(os.path.basename(new_image_path))[0]
            save_path = os.path.join(save_dir, f"{image_name}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        # Print match details
        print(f"\nRegion Matching Details:")
        print(f"Reference regions: {len(self.reference_features['regions'])}")
        print(f"Test regions: {len(details['test_regions'])}")
        print(f"Matches found: {len(details['matches'])}")
        
        for i, (ref_idx, test_idx, sim) in enumerate(details['matches']):
            ref_region = self.reference_features['regions'][ref_idx]
            test_region = details['test_regions'][test_idx]
            print(f"Match {i+1}: Ref region {ref_idx} -> Test region {test_idx}, Similarity: {sim:.3f}")
            print(f"  Ref activation: {ref_region['mean_activation']:.3f}, Test activation: {test_region['mean_activation']:.3f}")
        
        return similarity_score
