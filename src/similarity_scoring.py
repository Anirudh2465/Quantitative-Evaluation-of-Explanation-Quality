import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.segmentation import slic
try:
    from skimage.future import graph as rag_module
except ImportError:
    from skimage import graph as rag_module
from skimage import measure
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')


class SuperpixelGraphSimilarityScorer:
    """
    Computes similarity between Grad-CAM explanations using superpixel-based
    Region Adjacency Graphs (RAGs).

    Pipeline:
      1. Generate Grad-CAM heatmaps for reference and test images.
      2. Over-segment each image into SLIC superpixels.
      3. Build a RAG where each node is a superpixel with features
         (mean activation, mean color, centroid, area) and edges connect
         spatially adjacent superpixels.
      4. Filter to "active" subgraphs (superpixels whose mean activation
         exceeds the threshold).
      5. Compare the reference and test active subgraphs using:
         a. Node-level feature matching via the Hungarian algorithm.
         b. Structural similarity via edge overlap (Jaccard of adjacency).
         c. Global activation distribution alignment (histogram correlation).
      6. Return a composite similarity score in [0, 1].
    """

    def __init__(self, model_path, reference_image_path,
                 activation_threshold=0.5, n_superpixels=200, compactness=20):
        """
        Args:
            model_path: Path to trained DenseNet-121 .pth weights.
            reference_image_path: Path to the reference ultrasound image.
            activation_threshold: Minimum mean activation for a superpixel
                                  to be included in the active subgraph.
            n_superpixels: Target number of SLIC superpixels.
            compactness: SLIC compactness parameter (higher = more regular shapes).
        """
        self.model_path = model_path
        self.reference_image_path = reference_image_path
        self.activation_threshold = activation_threshold
        self.n_superpixels = n_superpixels
        self.compactness = compactness

        # Load model
        self.model = self._load_model()
        if self.model is None:
            raise ValueError("Failed to load model")

        # Target layer for Grad-CAM
        self.target_layer = self.model.features.denseblock4.denselayer16.conv2

        # Initialize GradCAM
        self.gradcam = GradCAM(model=self.model, target_layers=[self.target_layer])

        # Preprocessing transforms
        self.viz_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Extract reference superpixel graph
        self.reference_graph = self._build_graph_for_image(self.reference_image_path)
        print(f"Reference graph: {self.reference_graph['n_active']} active superpixels "
              f"out of {self.reference_graph['n_total']}")

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

    def _generate_gradcam(self, image_path):
        """Generate Grad-CAM heatmap for an image.

        Returns:
            img_np: (224, 224, 3) float32 array in [0, 1]
            cam:    (224, 224) float32 activation map in [0, 1]
        """
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = self.viz_transform(original_image).unsqueeze(0)

        targets = [ClassifierOutputTarget(0)]
        cam = self.gradcam(input_tensor=input_tensor, targets=targets)[0]

        img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        return img_np, cam

    # ------------------------------------------------------------------
    # Superpixel graph construction
    # ------------------------------------------------------------------

    def _compute_superpixels(self, image_uint8):
        """Compute SLIC superpixels on an 8-bit RGB image (224×224×3).

        Returns:
            segments: (224, 224) int array of superpixel labels (0-indexed).
        """
        segments = slic(
            image_uint8,
            n_segments=self.n_superpixels,
            compactness=self.compactness,
            sigma=1,
            start_label=0,
            channel_axis=2
        )
        return segments

    def _build_superpixel_features(self, image_uint8, cam, segments):
        """Compute per-superpixel features.

        Returns:
            sp_features: dict mapping superpixel_id → feature dict
        """
        sp_ids = np.unique(segments)
        sp_features = {}

        for sp_id in sp_ids:
            mask = segments == sp_id
            area = int(np.sum(mask))

            # Mean activation within this superpixel
            mean_activation = float(np.mean(cam[mask]))
            max_activation = float(np.max(cam[mask]))

            # Mean color (RGB)
            mean_rgb = np.mean(image_uint8[mask], axis=0).astype(np.float32)

            # Centroid (normalized to [0, 1])
            ys, xs = np.where(mask)
            centroid = np.array([np.mean(ys) / 224.0, np.mean(xs) / 224.0],
                                dtype=np.float32)

            sp_features[sp_id] = {
                'id': int(sp_id),
                'mask': mask,
                'area': area,
                'area_ratio': area / (224 * 224),
                'mean_activation': mean_activation,
                'max_activation': max_activation,
                'mean_rgb': mean_rgb,
                'centroid': centroid,
            }

        return sp_features

    def _build_adjacency(self, segments, sp_features):
        """Build adjacency list from the superpixel segment map.

        Two superpixels are adjacent if they share at least one pair of
        4-connected neighboring pixels across the boundary.

        Returns:
            adjacency: dict mapping sp_id → set of neighbor sp_ids
        """
        adjacency = {sp_id: set() for sp_id in sp_features}

        # Horizontal neighbors
        h_diff = segments[:, :-1] != segments[:, 1:]
        ys, xs = np.where(h_diff)
        for y, x in zip(ys, xs):
            a, b = int(segments[y, x]), int(segments[y, x + 1])
            adjacency[a].add(b)
            adjacency[b].add(a)

        # Vertical neighbors
        v_diff = segments[:-1, :] != segments[1:, :]
        ys, xs = np.where(v_diff)
        for y, x in zip(ys, xs):
            a, b = int(segments[y, x]), int(segments[y + 1, x])
            adjacency[a].add(b)
            adjacency[b].add(a)

        return adjacency

    def _extract_active_subgraph(self, sp_features, adjacency):
        """Filter to superpixels whose mean activation >= threshold.

        Returns:
            active_ids: sorted list of active superpixel IDs
            active_features: dict subset of sp_features
            active_adjacency: adjacency restricted to active IDs
        """
        active_ids = sorted([
            sp_id for sp_id, feat in sp_features.items()
            if feat['mean_activation'] >= self.activation_threshold
        ])

        # Fallback: if nothing passes, lower threshold to 70%
        if len(active_ids) == 0:
            fallback_thresh = self.activation_threshold * 0.7
            active_ids = sorted([
                sp_id for sp_id, feat in sp_features.items()
                if feat['mean_activation'] >= fallback_thresh
            ])
            if len(active_ids) == 0:
                # Last resort: take top-10 by activation
                ranked = sorted(sp_features.items(),
                                key=lambda kv: kv[1]['mean_activation'],
                                reverse=True)
                active_ids = sorted([sp_id for sp_id, _ in ranked[:10]])

        active_set = set(active_ids)
        active_features = {sp_id: sp_features[sp_id] for sp_id in active_ids}
        active_adjacency = {
            sp_id: adjacency[sp_id] & active_set
            for sp_id in active_ids
        }

        return active_ids, active_features, active_adjacency

    def _build_graph_for_image(self, image_path):
        """Full pipeline: image → Grad-CAM → superpixels → active subgraph.

        Returns a dict with all graph data needed for comparison.
        """
        img_np, cam = self._generate_gradcam(image_path)

        # Convert to uint8 for SLIC
        if img_np.max() <= 1.0:
            image_uint8 = (img_np * 255).astype(np.uint8)
        else:
            image_uint8 = img_np.astype(np.uint8)

        segments = self._compute_superpixels(image_uint8)
        sp_features = self._build_superpixel_features(image_uint8, cam, segments)
        adjacency = self._build_adjacency(segments, sp_features)

        active_ids, active_features, active_adjacency = \
            self._extract_active_subgraph(sp_features, adjacency)

        return {
            'image': img_np,
            'cam': cam,
            'segments': segments,
            'sp_features': sp_features,
            'adjacency': adjacency,
            'active_ids': active_ids,
            'active_features': active_features,
            'active_adjacency': active_adjacency,
            'n_active': len(active_ids),
            'n_total': len(sp_features),
        }

    # ------------------------------------------------------------------
    # Superpixel-level similarity
    # ------------------------------------------------------------------

    def _superpixel_feature_similarity(self, feat_a, feat_b):
        """Compute feature similarity between two individual superpixels.

        Combines:
          - Activation similarity  (40%)
          - Spatial IoU            (30%)
          - Centroid proximity     (15%)
          - Color similarity       (15%)
        """
        # Activation similarity
        act_sim = 1.0 - abs(feat_a['mean_activation'] - feat_b['mean_activation'])
        max_act_sim = 1.0 - abs(feat_a['max_activation'] - feat_b['max_activation'])
        activation_score = (act_sim + max_act_sim) / 2.0

        # Spatial IoU between superpixel masks
        intersection = np.logical_and(feat_a['mask'], feat_b['mask']).sum()
        union = np.logical_or(feat_a['mask'], feat_b['mask']).sum()
        iou = intersection / (union + 1e-6)

        # Centroid distance (exponential decay)
        centroid_dist = np.linalg.norm(feat_a['centroid'] - feat_b['centroid'])
        centroid_score = np.exp(-centroid_dist * 5.0)

        # Color similarity (normalized L2 in RGB space)
        color_dist = np.linalg.norm(feat_a['mean_rgb'] - feat_b['mean_rgb'])
        color_score = 1.0 - color_dist / (255.0 * np.sqrt(3))

        return (0.40 * activation_score +
                0.30 * iou +
                0.15 * centroid_score +
                0.15 * max(0.0, color_score))

    # ------------------------------------------------------------------
    # Graph-level similarity
    # ------------------------------------------------------------------

    def _node_matching_similarity(self, ref_graph, test_graph):
        """Hungarian-matched node similarity between active subgraphs.

        Returns:
            node_sim: mean similarity of optimally matched superpixels
            matches:  list of (ref_sp_id, test_sp_id, sim) tuples
        """
        ref_ids = ref_graph['active_ids']
        test_ids = test_graph['active_ids']

        if len(ref_ids) == 0 or len(test_ids) == 0:
            return 0.0, []

        # Build cost matrix
        sim_matrix = np.zeros((len(ref_ids), len(test_ids)))
        for i, r_id in enumerate(ref_ids):
            for j, t_id in enumerate(test_ids):
                sim_matrix[i, j] = self._superpixel_feature_similarity(
                    ref_graph['active_features'][r_id],
                    test_graph['active_features'][t_id]
                )

        cost_matrix = 1.0 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_sims = []
        matches = []
        for r_idx, t_idx in zip(row_ind, col_ind):
            sim = sim_matrix[r_idx, t_idx]
            matched_sims.append(sim)
            matches.append((ref_ids[r_idx], test_ids[t_idx], sim))

        avg_sim = np.mean(matched_sims) if matched_sims else 0.0

        # Penalize unmatched nodes
        n_unmatched = abs(len(ref_ids) - len(test_ids))
        n_max = max(len(ref_ids), len(test_ids))
        penalty = 0.3 * n_unmatched / n_max
        node_sim = avg_sim * (1.0 - penalty)

        return node_sim, matches

    def _structural_similarity(self, ref_graph, test_graph, matches):
        """Compare graph topology by checking how many edges are preserved.

        For each matched pair of nodes (ref_a, test_a) and (ref_b, test_b):
          - If ref_a and ref_b are adjacent in the reference graph AND
            test_a and test_b are adjacent in the test graph, the edge is
            preserved.

        Returns:
            structural_sim: fraction of reference edges preserved in [0, 1]
        """
        if len(matches) < 2:
            return 1.0  # Trivially satisfied

        # Build match mapping: ref_sp → test_sp
        ref_to_test = {r_id: t_id for r_id, t_id, _ in matches}

        ref_adj = ref_graph['active_adjacency']
        test_adj = test_graph['active_adjacency']

        # Count reference edges between matched nodes
        ref_edges = set()
        matched_ref_ids = set(ref_to_test.keys())
        for r_id in matched_ref_ids:
            for r_neighbor in ref_adj.get(r_id, set()):
                if r_neighbor in matched_ref_ids:
                    edge = tuple(sorted([r_id, r_neighbor]))
                    ref_edges.add(edge)

        if len(ref_edges) == 0:
            return 1.0  # No edges to compare

        # Check how many are preserved in test graph
        preserved = 0
        for r_a, r_b in ref_edges:
            t_a = ref_to_test[r_a]
            t_b = ref_to_test[r_b]
            if t_b in test_adj.get(t_a, set()):
                preserved += 1

        return preserved / len(ref_edges)

    def _activation_distribution_similarity(self, ref_graph, test_graph):
        """Compare global activation distributions using histogram correlation.

        Uses the full CAM (not just active superpixels) to measure overall
        agreement in where the model focuses.

        Returns:
            hist_sim: correlation in [0, 1]
        """
        n_bins = 32

        ref_hist = np.histogram(ref_graph['cam'].ravel(), bins=n_bins,
                                range=(0, 1))[0].astype(np.float32)
        test_hist = np.histogram(test_graph['cam'].ravel(), bins=n_bins,
                                 range=(0, 1))[0].astype(np.float32)

        # Normalize
        ref_hist /= (ref_hist.sum() + 1e-8)
        test_hist /= (test_hist.sum() + 1e-8)

        # Histogram correlation (cv2 returns value in [-1, 1])
        corr = cv2.compareHist(ref_hist, test_hist, cv2.HISTCMP_CORREL)
        return max(0.0, (corr + 1.0) / 2.0)  # Map to [0, 1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_focused_similarity(self, new_image_path):
        """Compute superpixel-graph-based similarity score.

        Returns:
            similarity_score: float in [0, 1]
            details: dict with intermediate data for visualization
        """
        test_graph = self._build_graph_for_image(new_image_path)
        print(f"Test graph: {test_graph['n_active']} active superpixels "
              f"out of {test_graph['n_total']}")

        # Component 1: Node-level matching (50% weight)
        node_sim, matches = self._node_matching_similarity(
            self.reference_graph, test_graph
        )

        # Component 2: Structural / edge preservation (30% weight)
        struct_sim = self._structural_similarity(
            self.reference_graph, test_graph, matches
        )

        # Component 3: Activation distribution alignment (20% weight)
        dist_sim = self._activation_distribution_similarity(
            self.reference_graph, test_graph
        )

        similarity_score = (0.50 * node_sim +
                            0.30 * struct_sim +
                            0.20 * dist_sim)

        details = {
            'test_graph': test_graph,
            'matches': matches,
            'node_similarity': node_sim,
            'structural_similarity': struct_sim,
            'distribution_similarity': dist_sim,
            'test_cam': test_graph['cam'],
            'test_image': test_graph['image'],
        }

        return similarity_score, details

    def visualize_focused_comparison(self, new_image_path, save_path=None):
        """Visualize the superpixel graph comparison.

        Shows 4 panels:
          1. Test image
          2. SLIC superpixel boundaries
          3. Active superpixels highlighted (with matched edges)
          4. Side-by-side CAM overlay
        """
        similarity_score, details = self.compute_focused_similarity(new_image_path)
        image_name = os.path.splitext(os.path.basename(new_image_path))[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # --- Panel 1: Test image ---
        test_img = details['test_image']
        if test_img.max() <= 1.0:
            test_img_disp = (test_img * 255).astype(np.uint8)
        else:
            test_img_disp = test_img.astype(np.uint8)

        axes[0].imshow(test_img_disp)
        axes[0].set_title(f'Image {image_name}', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # --- Panel 2: Active superpixels highlighted ---
        overlay = test_img_disp.copy().astype(np.float32) / 255.0
        test_g = details['test_graph']
        for sp_id in test_g['active_ids']:
            mask = test_g['active_features'][sp_id]['mask']
            activation = test_g['active_features'][sp_id]['mean_activation']
            # Color intensity proportional to activation
            green_channel = np.array([0, activation, 0])
            overlay[mask] = overlay[mask] * 0.5 + green_channel * 0.5

        # Draw superpixel boundaries
        segments = test_g['segments']
        boundary_mask = np.zeros(segments.shape, dtype=bool)
        boundary_mask[:, :-1] |= segments[:, :-1] != segments[:, 1:]
        boundary_mask[:-1, :] |= segments[:-1, :] != segments[1:, :]
        overlay[boundary_mask] = [1.0, 1.0, 1.0]  # White boundaries

        axes[1].imshow(np.clip(overlay, 0, 1))
        axes[1].set_title(
            f'Active Superpixels (score={similarity_score:.3f})',
            fontsize=12, fontweight='bold'
        )
        axes[1].axis('off')

        plt.tight_layout()

        # Save
        if save_path is None:
            save_dir = "gradcam_outputs"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{image_name}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

        plt.show()

        # Print detailed report
        print(f"\n{'='*50}")
        print(f"Superpixel Graph Similarity Report")
        print(f"{'='*50}")
        print(f"Reference active superpixels : {self.reference_graph['n_active']}")
        print(f"Test active superpixels      : {test_g['n_active']}")
        print(f"Matched pairs                : {len(details['matches'])}")
        print(f"")
        print(f"Node matching similarity     : {details['node_similarity']:.4f}  (50% weight)")
        print(f"Structural similarity        : {details['structural_similarity']:.4f}  (30% weight)")
        print(f"Distribution similarity      : {details['distribution_similarity']:.4f}  (20% weight)")
        print(f"")
        print(f"Final similarity score       : {similarity_score:.4f}")
        print(f"{'='*50}")

        for i, (ref_id, test_id, sim) in enumerate(details['matches'][:10]):
            ref_feat = self.reference_graph['active_features'][ref_id]
            test_feat = test_g['active_features'][test_id]
            print(f"  Match {i+1}: SP-{ref_id} -> SP-{test_id}  "
                  f"sim={sim:.3f}  "
                  f"act={ref_feat['mean_activation']:.2f}->{test_feat['mean_activation']:.2f}")

        if len(details['matches']) > 10:
            print(f"  ... and {len(details['matches']) - 10} more matches")

        return similarity_score


# Backward-compatible alias
FocusedGradCAMSimilarityScorer = SuperpixelGraphSimilarityScorer
