"""
TCAV (Testing with Concept Activation Vectors) scoring module.

Quantifies how much a human-understandable *concept* (e.g. "Caliper present")
influences a trained model's predictions.  This provides a complementary
metric to pixel-level saliency methods like Grad-CAM.

Pipeline
--------
1.  Collect concept-positive and random images.
2.  Extract bottleneck activations for both sets.
3.  Train a linear classifier (CAV) to separate them.
4.  Compute directional derivatives of the model output w.r.t. the CAV.
5.  TCAV score = fraction of test images with positive sensitivity.

References
----------
Kim et al., "Interpretability Beyond Feature Attribution: Quantitative Testing
with Concept Activation Vectors (TCAV)", ICML 2018.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class LowCAVAccuracyError(ValueError):
    """Exception raised when CAV validation accuracy is below the required threshold."""
    def __init__(self, message, mean_cav_accuracy=None):
        super().__init__(message)
        self.mean_cav_accuracy = mean_cav_accuracy


class TCAVFloat(float):
    """Custom float subclass that carries TCAV statistical metadata while
    remaining mathematically compatible with standard float operations.
    """
    def __new__(cls, value, p_value=None, is_significant=None, ci=None, details=None):
        val = float(value) if value is not None else float('nan')
        obj = super().__new__(cls, val)
        obj.p_value = p_value
        obj.is_significant = is_significant
        obj.ci = ci  # tuple: (ci_low, ci_high)
        obj.details = details or {}
        return obj


# ──────────────────────────────────────────────
# Model helpers (shared architecture with the rest of the codebase)
# ──────────────────────────────────────────────

def _build_model():
    """Build a DenseNet-121 with the same classifier head used during training."""
    model = models.densenet121(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model


def _load_model(model_path, device="cpu"):
    """Load trained weights into the standard architecture."""
    model = _build_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


# ──────────────────────────────────────────────
# Preprocessing transform (matches training validation transform)
# ──────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# TCAVScorer
# ──────────────────────────────────────────────

class TCAVScorer:
    """Compute TCAV scores for a trained biomarker classifier.

    Parameters
    ----------
    model_path : str
        Path to the trained DenseNet-121 ``.pth`` file.
    biomarker_name : str
        Name of the biomarker (column in the CSV).
    concept_csv : str
        CSV with at least columns ``['Image ID', biomarker_name]``.
    img_dir : str
        Directory containing ``{image_id}.jpg`` files.
    device : str or torch.device
        ``'cpu'`` or ``'cuda'``.
    activation_threshold : float
        Minimum CAV classifier accuracy to consider a concept learnable.
    """

    # Target layer — default bottleneck used for Grad-CAM
    TARGET_LAYER_NAME = "features.denseblock4.denselayer16.conv2"

    def __init__(
        self,
        model_path,
        biomarker_name,
        concept_csv,
        img_dir,
        device="cpu",
        activation_threshold=0.55,
        concept_dir=None,
        target_layer_name=TARGET_LAYER_NAME,
        classifier_type="logistic_regression",
        cav_accuracy_threshold=0.80,
        smoothgrad=True,
        smoothgrad_samples=20,
        smoothgrad_noise=0.15,
    ):
        self.model_path = model_path
        self.biomarker_name = biomarker_name
        self.concept_csv = concept_csv
        self.img_dir = img_dir
        self.device = torch.device(device)
        self.activation_threshold = activation_threshold
        self.concept_dir = concept_dir
        self.target_layer_name = target_layer_name
        self.classifier_type = classifier_type
        self.cav_accuracy_threshold = cav_accuracy_threshold
        self.smoothgrad = smoothgrad
        self.smoothgrad_samples = smoothgrad_samples
        self.smoothgrad_noise = smoothgrad_noise

        # Load model
        self.model = _load_model(model_path, device=self.device)

        # Resolve target layer
        self.target_layer = self._resolve_layer(self.target_layer_name)

        # Load label data
        df = pd.read_csv(concept_csv)
        df.columns = [c.strip() for c in df.columns]
        self.label_df = df

        # Split into concept-positive and concept-negative image IDs
        self.concept_positive_ids = df.loc[
            df[biomarker_name] == 1, "Image ID"
        ].tolist()
        self.concept_negative_ids = df.loc[
            df[biomarker_name] == 0, "Image ID"
        ].tolist()

        print(
            f"[TCAV] {biomarker_name}: "
            f"{len(self.concept_positive_ids)} concept+ / "
            f"{len(self.concept_negative_ids)} concept- images"
        )
        self.activation_cache = {}
        self.grad_cache = {}

        # Resolve folders for custom training
        if concept_dir:
            self.present_dir = os.path.join(concept_dir, "present")
            self.absent_dir = os.path.join(concept_dir, "absent")
            if not os.path.exists(self.absent_dir):
                self.absent_dir = os.path.join(concept_dir, "absemt")
            print(f"[TCAV] Custom concept training folders resolved:")
            print(f"  - Present: {self.present_dir}")
            print(f"  - Absent/Absemt: {self.absent_dir}")

        # Concept transform specifically for arbitrary sizes / manual crops
        self.concept_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ---- internal helpers ------------------------------------------------

    def _resolve_layer(self, dotted_name):
        """Walk the module tree to find a layer by dotted name."""
        module = self.model
        for part in dotted_name.split("."):
            module = getattr(module, part)
        return module

    def _load_image(self, image_id):
        """Load a single image as a normalised tensor."""
        path = os.path.join(self.img_dir, f"{int(image_id)}.jpg")
        img = Image.open(path).convert("RGB")
        return _transform(img)

    # ---- activation extraction -------------------------------------------

    def _extract_activations(self, image_ids, batch_size=32):
        """Return flattened activations at the target layer for *image_ids*.

        Uses an internal activation cache to avoid repeated forward passes of DenseNet.
        """
        needed_ids = [img_id for img_id in image_ids if img_id not in self.activation_cache]
        
        if needed_ids:
            captured = {}
            def hook_fn(_module, _input, output):
                captured["act"] = output

            handle = self.target_layer.register_forward_hook(hook_fn)
            try:
                for start in range(0, len(needed_ids), batch_size):
                    batch_ids = needed_ids[start : start + batch_size]
                    tensors = []
                    valid_ids = []
                    for img_id in batch_ids:
                        try:
                            tensors.append(self._load_image(img_id))
                            valid_ids.append(img_id)
                        except FileNotFoundError:
                            continue
                    if not tensors:
                        continue

                    batch = torch.stack(tensors).to(self.device)
                    with torch.no_grad():
                        self.model(batch)

                    act = captured["act"]  # (B, C, H, W)
                    act = torch.nn.functional.adaptive_avg_pool2d(act, 1)
                    act = act.view(act.size(0), -1)  # (B, C)
                    act_np = act.cpu().numpy()
                    
                    for i, img_id in enumerate(valid_ids):
                        self.activation_cache[img_id] = act_np[i]
            finally:
                handle.remove()
                
        # Retrieve all requested activations from cache
        acts = []
        for img_id in image_ids:
            if img_id in self.activation_cache:
                acts.append(self.activation_cache[img_id])
                
        if not acts:
            return np.empty((0, 0))
            
        return np.stack(acts)

    def _extract_folder_activations(self, folder_path, batch_size=32):
        """Extract bottleneck activations for all images in the folder.
        
        Handles manually cropped images of varying sizes by resizing and normalizing.
        """
        if not folder_path or not os.path.isdir(folder_path):
            return np.empty((0, 0))

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        img_paths = []
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                img_paths.append(os.path.join(folder_path, file))

        if not img_paths:
            return np.empty((0, 0))

        acts = []
        captured = {}

        def hook_fn(_module, _input, output):
            captured["act"] = output

        handle = self.target_layer.register_forward_hook(hook_fn)
        try:
            for start in range(0, len(img_paths), batch_size):
                batch_paths = img_paths[start : start + batch_size]
                tensors = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        tensors.append(self.concept_transform(img))
                    except Exception as e:
                        print(f"[TCAV] Error loading folder image '{p}': {e}")
                        continue
                if not tensors:
                    continue

                batch = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    self.model(batch)

                act = captured["act"]  # (B, C, H, W)
                act = torch.nn.functional.adaptive_avg_pool2d(act, 1)
                act = act.view(act.size(0), -1)  # (B, C)
                acts.append(act.cpu().numpy())
        finally:
            handle.remove()

        if not acts:
            return np.empty((0, 0))

        return np.concatenate(acts, axis=0)

    # ---- CAV training ----------------------------------------------------

    def _train_cav(self, concept_acts, random_acts):
        """Train a classifier separating concept from random activations.

        Returns
        -------
        cav : np.ndarray, shape (n_features,)
            The Concept Activation Vector (unit-normed weight vector).
        accuracy : float
            Classifier accuracy on a held-out split.
        """
        X = np.concatenate([concept_acts, random_acts], axis=0)
        y = np.concatenate([
            np.ones(len(concept_acts)),
            np.zeros(len(random_acts)),
        ])

        # Stratified split for evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=None
        )

        if self.classifier_type == "sgd_hinge":
            clf = SGDClassifier(
                loss="hinge",
                max_iter=1000,
                tol=1e-4,
                random_state=None,
            )
        else:  # default 'logistic_regression'
            clf = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                random_state=None,
            )

        clf.fit(X_train, y_train)
        accuracy = clf.score(X_val, y_val)

        # CAV = unit-normed weight vector pointing toward the concept
        cav = clf.coef_[0].copy()
        norm = np.linalg.norm(cav)
        if norm > 0:
            cav /= norm

        return cav, accuracy

    # ---- directional derivative ------------------------------------------

    def _compute_gradients_smoothgrad(self, image_id, target_class=1, num_samples=20, noise_level=0.15):
        """Compute activation-level gradients using SmoothGrad to reduce noise."""
        tensor = self._load_image(image_id).unsqueeze(0).to(self.device)
        
        captured = {}
        def hook_fn(_module, _input, output):
            captured["act"] = output

        handle = self.target_layer.register_forward_hook(hook_fn)
        grad_accumulator = None
        
        try:
            for _ in range(num_samples):
                if num_samples > 1 and noise_level > 0:
                    noise = torch.randn_like(tensor) * noise_level
                    noisy_tensor = tensor + noise
                else:
                    noisy_tensor = tensor.clone()
                    
                noisy_tensor.requires_grad_(False)
                self.model.zero_grad()
                
                # Forward pass
                output = self.model(noisy_tensor)
                
                # Explain target class probability (Sigmoid output)
                if target_class == 0:
                    score = 1.0 - output
                else:
                    score = output
                
                act = captured["act"]
                
                grad = torch.autograd.grad(
                    outputs=score,
                    inputs=act,
                    retain_graph=True,
                    create_graph=False,
                )[0]  # (1, C, H, W)
                
                if grad_accumulator is None:
                    grad_accumulator = grad.clone()
                else:
                    grad_accumulator += grad
                    
            mean_grad = grad_accumulator / num_samples
            return mean_grad  # (1, C, H, W)
        finally:
            handle.remove()

    def _compute_directional_sensitivity_for_image(self, image_id, cav):
        """Compute directional sensitivity (both global and spatial) for an image ID."""
        target_row = self.label_df[self.label_df["Image ID"] == image_id]
        target_class = int(target_row[self.biomarker_name].values[0]) if len(target_row) > 0 else 1

        # Cache the GRADIENT itself (which is independent of CAV), not the dot product!
        cache_key = (image_id, self.target_layer_name, target_class)
        if cache_key in self.grad_cache:
            grad_pooled, mean_grad = self.grad_cache[cache_key]
        else:
            if self.smoothgrad:
                mean_grad = self._compute_gradients_smoothgrad(
                    image_id, target_class=target_class, num_samples=self.smoothgrad_samples, noise_level=self.smoothgrad_noise
                )
            else:
                mean_grad = self._compute_gradients_smoothgrad(
                    image_id, target_class=target_class, num_samples=1, noise_level=0.0
                )
            grad_pooled = torch.nn.functional.adaptive_avg_pool2d(mean_grad, 1)
            grad_pooled = grad_pooled.view(-1).cpu().numpy()  # (C,)
            self.grad_cache[cache_key] = (grad_pooled, mean_grad)
            
        # 1. Global sensitivity: dot product with CAV.
        global_sens = float(np.dot(grad_pooled, cav))
        
        # 2. Spatial sensitivity map: pixel-by-pixel dot product with CAV.
        grad_np = mean_grad[0].cpu().numpy()  # (C, H, W)
        spatial_sens = np.tensordot(grad_np, cav, axes=(0, 0))  # (H, W)
        
        # 3. Normalized sensitivity (cosine similarity) for Relative TCAV:
        grad_norm = np.linalg.norm(grad_pooled)
        cosine_sim = float(np.dot(grad_pooled, cav) / (grad_norm + 1e-8)) if grad_norm > 0 else 0.0
        
        result = {
            "global_sensitivity": global_sens,
            "spatial_sensitivity": spatial_sens,
            "cosine_similarity": cosine_sim,
            "grad_norm": float(grad_norm)
        }
        
        return result

    def _compute_directional_sensitivity(self, image_id, cav):
        """Deprecated: use _compute_directional_sensitivity_for_image instead."""
        res = self._compute_directional_sensitivity_for_image(image_id, cav)
        return res["global_sensitivity"]

    # ---- Spatial TCAV Heatmap --------------------------------------------

    def generate_concept_heatmap(self, image_id, cav):
        """Generate a spatial concept heatmap showing where the concept activates the model.
        
        Returns
        -------
        heatmap : np.ndarray (224, 224)
            Normalized concept heatmap in [0, 1].
        spatial_sens : np.ndarray (H, W)
            Raw pixel-by-pixel directional sensitivities.
        """
        sens_dict = self._compute_directional_sensitivity_for_image(image_id, cav)
        spatial_sens = sens_dict["spatial_sensitivity"]  # (H, W)
        
        # Apply ReLU to focus on positive concept influence
        heatmap = np.maximum(spatial_sens, 0)
        
        # Normalize to [0, 1]
        h_max = heatmap.max()
        h_min = heatmap.min()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)
        else:
            heatmap = np.zeros_like(heatmap)
            
        # Resize to input resolution (224, 224)
        heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        return heatmap_resized, spatial_sens

    # ---- TCAV score ------------------------------------------------------

    def compute_tcav_score(self, test_image_ids, concept_ids=None,
                           random_ids=None, max_concept=None, max_random=None,
                           concept_acts=None, random_acts=None):
        """Compute a single TCAV score.

        Parameters
        ----------
        test_image_ids : list
            Image IDs to evaluate sensitivity on.
        concept_ids : list or None
            Override concept-positive IDs.  Defaults to self.concept_positive_ids.
        random_ids : list or None
            Override random IDs.  Defaults to self.concept_negative_ids.
        max_concept, max_random : int or None
            Cap the number of concept / random samples (for speed).
        concept_acts : np.ndarray or None
            Pre-extracted concept-positive activations. If provided, overrides concept_ids.
        random_acts : np.ndarray or None
            Pre-extracted concept-negative activations. If provided, overrides random_ids.

        Returns
        -------
        tcav_score : float
            Fraction of test images with positive sensitivity.
        cav_accuracy : float
            Accuracy of the CAV linear classifier.
        sensitivities : list
            Raw sensitivities of the test images.
        cosine_sims : list
            Cosine similarities of the test images.
        cav : np.ndarray
            The trained Concept Activation Vector.
        """
        # If activations are not provided, we extract them
        if concept_acts is None:
            if concept_ids is None:
                concept_ids = list(self.concept_positive_ids)
            # Remove test images to avoid leakage
            test_set = set(test_image_ids)
            concept_ids = [i for i in concept_ids if i not in test_set]
            if max_concept and len(concept_ids) > max_concept:
                rng = np.random.default_rng()
                concept_ids = rng.choice(concept_ids, size=max_concept,
                                         replace=False).tolist()
            if len(concept_ids) < 5:
                print("[TCAV] WARNING: Not enough concept images. Returning baseline.")
                return 0.5, 0.0, [0.0]*len(test_image_ids), [0.0]*len(test_image_ids), np.zeros(1)
            concept_acts = self._extract_activations(concept_ids)

        if random_acts is None:
            if random_ids is None:
                random_ids = list(self.concept_negative_ids)
            # Remove test images to avoid leakage
            test_set = set(test_image_ids)
            random_ids = [i for i in random_ids if i not in test_set]
            if max_random and len(random_ids) > max_random:
                rng = np.random.default_rng()
                random_ids = rng.choice(random_ids, size=max_random,
                                        replace=False).tolist()
            if len(random_ids) < 5:
                print("[TCAV] WARNING: Not enough random images. Returning baseline.")
                return 0.5, 0.0, [0.0]*len(test_image_ids), [0.0]*len(test_image_ids), np.zeros(1)
            random_acts = self._extract_activations(random_ids)

        if concept_acts.shape[0] == 0 or random_acts.shape[0] == 0:
            return 0.5, 0.0, [0.0]*len(test_image_ids), [0.0]*len(test_image_ids), np.zeros(1)

        # Train CAV
        cav, accuracy = self._train_cav(concept_acts, random_acts)

        # Compute directional sensitivity for each test image
        positive_count = 0
        total_count = 0
        sensitivities = []
        cosine_sims = []

        for img_id in test_image_ids:
            try:
                sens_dict = self._compute_directional_sensitivity_for_image(img_id, cav)
                sens = sens_dict["global_sensitivity"]
                cosine_sim = sens_dict["cosine_similarity"]
                
                sensitivities.append(sens)
                cosine_sims.append(cosine_sim)
                
                if sens > 0:
                    positive_count += 1
                total_count += 1
            except Exception as e:
                print(f"[TCAV] Sensitivity error for image {img_id}: {e}")
                continue

        tcav_score = positive_count / total_count if total_count > 0 else 0.5
        return tcav_score, accuracy, sensitivities, cosine_sims, cav

    def compute_tcav_with_statistical_testing(
        self, test_image_ids, n_random_runs=50, significance_level=0.05
    ):
        """Run TCAV multiple times with shuffled random sets and test significance.

        Parameters
        ----------
        test_image_ids : list
            Image IDs to evaluate.
        n_random_runs : int
            Number of random baseline runs for the statistical test.
        significance_level : float
            p-value threshold for significance.

        Returns
        -------
        mean_tcav_score : float
            Mean TCAV score across all runs.
        p_value : float
            One-sided p-value (concept TCAV > 0.5).
        is_significant : bool
            Whether the concept is statistically significant.
        details : dict
            Per-run scores, CAV accuracies, etc.
        """
        all_ids = list(self.label_df["Image ID"])
        concept_ids = list(self.concept_positive_ids)
        negative_ids = list(self.concept_negative_ids)
        rng = np.random.default_rng()

        concept_scores = []
        random_scores = []
        cav_accuracies = []

        run_mean_sensitivities = []
        run_mean_abs_sensitivities = []
        run_mean_cosine_similarities = []

        best_cav_acc = -1.0
        best_cav = None

        # Load folder-based activations if available
        folder_pos_acts = None
        folder_neg_acts = None
        if hasattr(self, "concept_dir") and self.concept_dir:
            folder_pos_acts = self._extract_folder_activations(self.present_dir)
            folder_neg_acts = self._extract_folder_activations(self.absent_dir)
            if folder_pos_acts.shape[0] == 0:
                print("[TCAV] WARNING: No positive concept images found in folder. Fallback to CSV.")
                folder_pos_acts = None
            else:
                print(f"[TCAV] Loaded {folder_pos_acts.shape[0]} positive activations from folder.")
            if folder_neg_acts.shape[0] > 0:
                print(f"[TCAV] Loaded {folder_neg_acts.shape[0]} negative activations from folder.")
            else:
                folder_neg_acts = None

        for run in range(n_random_runs):
            # --- Concept run: real concept vs shuffled negative/random set ---
            if folder_pos_acts is not None:
                if folder_neg_acts is not None and folder_neg_acts.shape[0] >= 5:
                    idx = rng.choice(
                        folder_neg_acts.shape[0],
                        size=min(folder_neg_acts.shape[0], folder_pos_acts.shape[0]),
                        replace=False,
                    )
                    shuffled_neg_acts = folder_neg_acts[idx]
                else:
                    shuffled_neg_ids = rng.choice(
                        negative_ids,
                        size=min(len(negative_ids), folder_pos_acts.shape[0]),
                        replace=False,
                    ).tolist()
                    shuffled_neg_acts = self._extract_activations(shuffled_neg_ids)

                score_c, acc_c, sens_c, cos_c, cav = self.compute_tcav_score(
                    test_image_ids,
                    concept_acts=folder_pos_acts,
                    random_acts=shuffled_neg_acts,
                )
            else:
                shuffled_neg = rng.choice(
                    negative_ids,
                    size=min(len(negative_ids), len(concept_ids)),
                    replace=False,
                ).tolist()

                score_c, acc_c, sens_c, cos_c, cav = self.compute_tcav_score(
                    test_image_ids,
                    concept_ids=concept_ids,
                    random_ids=shuffled_neg,
                )
            
            concept_scores.append(score_c)
            cav_accuracies.append(acc_c)

            if sens_c:
                run_mean_sensitivities.append(np.mean(sens_c))
                run_mean_abs_sensitivities.append(np.mean(np.abs(sens_c)))
                run_mean_cosine_similarities.append(np.mean(cos_c))

            if acc_c > best_cav_acc:
                best_cav_acc = acc_c
                best_cav = cav

            # --- Random run: random "concept" vs random "negative" ---
            # Sample random sets matching the concept positive/negative size (capped to 100 for safety)
            size_to_sample = min(len(concept_ids) if folder_pos_acts is None else folder_pos_acts.shape[0], 100)
            shuffled_all = rng.permutation(all_ids).tolist()
            rand_concept = shuffled_all[:size_to_sample]
            rand_negative = shuffled_all[size_to_sample : 2 * size_to_sample]

            try:
                score_r, _, _, _, _ = self.compute_tcav_score(
                    test_image_ids,
                    concept_ids=rand_concept,
                    random_ids=rand_negative,
                )
            except LowCAVAccuracyError:
                score_r = 0.5

            random_scores.append(score_r)

        concept_scores = np.array(concept_scores)
        random_scores = np.array(random_scores)

        mean_tcav_score = float(np.mean(concept_scores))
        mean_cav_accuracy = float(np.mean(cav_accuracies))

        # Enforce accuracy threshold check on the average accuracy across all runs
        if mean_cav_accuracy < self.cav_accuracy_threshold:
            raise LowCAVAccuracyError(
                f"Mean CAV validation accuracy {mean_cav_accuracy:.4f} is below threshold {self.cav_accuracy_threshold:.4f}",
                mean_cav_accuracy=mean_cav_accuracy
            )

        # Swap t-test for Mann-Whitney U test (non-parametric)
        p_value = 1.0
        u_stat = 0.0
        if len(concept_scores) > 1 and len(random_scores) > 1:
            try:
                res = stats.mannwhitneyu(concept_scores, random_scores, alternative='greater')
                p_value = res.pvalue
                u_stat = res.statistic
            except Exception as e:
                print(f"[TCAV] Mann-Whitney U test error: {e}")
                p_value = 0.0 if np.mean(concept_scores) > np.mean(random_scores) else 1.0

        is_significant = p_value < significance_level

        # Calculate 95% Confidence Interval for the mean TCAV score
        std_err = stats.sem(concept_scores) if len(concept_scores) > 1 else 0.0
        if std_err > 0:
            ci_low, ci_high = stats.t.interval(0.95, df=len(concept_scores)-1, loc=mean_tcav_score, scale=std_err)
        else:
            ci_low, ci_high = mean_tcav_score, mean_tcav_score

        # Calculate Relative sensitivity metrics
        mean_sens = float(np.mean(run_mean_sensitivities)) if run_mean_sensitivities else 0.0
        mean_abs_sens = float(np.mean(run_mean_abs_sensitivities)) if run_mean_abs_sensitivities else 0.0
        rel_sens = float(np.mean(run_mean_cosine_similarities)) if run_mean_cosine_similarities else 0.0

        details = {
            "concept_scores": concept_scores.tolist(),
            "random_scores": random_scores.tolist(),
            "cav_accuracies": cav_accuracies,
            "mean_cav_accuracy": mean_cav_accuracy,
            "u_stat": float(u_stat),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "mean_sensitivity": mean_sens,
            "mean_abs_sensitivity": mean_abs_sens,
            "relative_sensitivity": rel_sens,
            "cav": best_cav
        }

        return mean_tcav_score, p_value, is_significant, details

    # ---- multi-layer profiling -------------------------------------------

    def profile_multi_layers(self, test_image_ids, layers_list, n_random_runs=50, significance_level=0.05):
        """Evaluate TCAV scores across multiple target layers to find where the concept is learned.
        
        Parameters
        ----------
        test_image_ids : list
            IDs of test images.
        layers_list : list of str
            List of dotted layer paths (e.g. ['features.denseblock1.denselayer6.conv2', ...]).
            
        Returns
        -------
        results : dict
            Mapping layer name -> dict of results (mean_score, p_value, is_significant, mean_cav_accuracy)
        """
        profile_results = {}
        original_layer_name = self.target_layer_name
        for layer_name in layers_list:
            print(f"\n[TCAV] Profiling layer: {layer_name}")
            try:
                # Switch target layer
                self.target_layer_name = layer_name
                self.target_layer = self._resolve_layer(layer_name)
                # Clear caches because activations and gradients are layer-specific!
                self.activation_cache.clear()
                self.grad_cache.clear()
                
                mean_score, p_value, is_significant, details = self.compute_tcav_with_statistical_testing(
                    test_image_ids=test_image_ids,
                    n_random_runs=n_random_runs,
                    significance_level=significance_level
                )
                profile_results[layer_name] = {
                    "mean_score": mean_score,
                    "p_value": p_value,
                    "is_significant": is_significant,
                    "mean_cav_accuracy": details["mean_cav_accuracy"],
                    "ci_low": details.get("ci_low"),
                    "ci_high": details.get("ci_high"),
                    "mean_sensitivity": details.get("mean_sensitivity"),
                    "mean_abs_sensitivity": details.get("mean_abs_sensitivity"),
                    "relative_sensitivity": details.get("relative_sensitivity"),
                    "status": "success"
                }
            except LowCAVAccuracyError as e:
                print(f"[TCAV] Layer {layer_name} failed accuracy threshold: {e}")
                profile_results[layer_name] = {
                    "status": "failed",
                    "error": str(e)
                }
            except Exception as e:
                print(f"[TCAV] Error profiling layer {layer_name}: {e}")
                profile_results[layer_name] = {
                    "status": "error",
                    "error": str(e)
                }
        # Restore original layer
        self.target_layer_name = original_layer_name
        self.target_layer = self._resolve_layer(original_layer_name)
        self.activation_cache.clear()
        self.grad_cache.clear()
        return profile_results


# ──────────────────────────────────────────────
# Convenience function (matches evaluate_cam_comprehensive interface)
# ──────────────────────────────────────────────

def evaluate_tcav(biomarker, image, img_dir, biomarker_csv, device="cpu",
                  n_random_runs=50, concept_dir=None,
                  target_layer_name="features.denseblock4.denselayer16.conv2",
                  classifier_type="logistic_regression",
                  cav_accuracy_threshold=0.80,
                  smoothgrad=True, smoothgrad_samples=20, smoothgrad_noise=0.15):
    """Evaluate TCAV for a single image and biomarker.

    Parameters
    ----------
    biomarker : str
        Biomarker name (e.g. ``'Zoom'``).
    image : str
        Path to the test image file.
    img_dir : str
        Directory containing all dataset images.
    biomarker_csv : str
        CSV file with ``['Image ID', biomarker]`` columns.
    device : str
        ``'cpu'`` or ``'cuda'``.
    n_random_runs : int
        Number of random runs for statistical testing.
    concept_dir : str or None
        Path to directory containing custom concept folders 'present/' and 'absent/'.
    target_layer_name : str
        Path to layer in model to extract activations from.
    classifier_type : str
        Classifier type: 'logistic_regression' or 'sgd_hinge'.
    cav_accuracy_threshold : float
        Accuracy threshold below which TCAV score is aborted.
    smoothgrad : bool
        Whether to use SmoothGrad denoising.
    smoothgrad_samples : int
        Number of samples for SmoothGrad.
    smoothgrad_noise : float
        Noise standard deviation for SmoothGrad.

    Returns
    -------
    tcav_score : TCAVFloat
        TCAV score in [0, 1] carrying statistical metadata.
    """
    # Resolve model path via config (new structure) with legacy fallbacks.
    import sys as _sys
    _proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj not in _sys.path:
        _sys.path.insert(0, _proj)
    try:
        import config as _cfg
        model_path = _cfg.model_path(biomarker)
    except Exception:
        model_path = f"{biomarker}_model.pth"

    if not os.path.exists(model_path):
        for _cand in [f"{biomarker}_model.pth", f"{biomarker}.pth", f"checkpoint_{biomarker}.pth"]:
            if os.path.exists(_cand):
                model_path = _cand
                break
        else:
            print(f"[TCAV] Model not found for {biomarker}. Returning 0.5.")
            return TCAVFloat(0.5, p_value=1.0, is_significant=False, ci=(0.5, 0.5))

    basename = os.path.splitext(os.path.basename(image))[0]
    try:
        test_image_id = int(basename)
    except ValueError:
        print(f"[TCAV] Cannot parse image ID from '{image}'. Returning 0.5.")
        return TCAVFloat(0.5, p_value=1.0, is_significant=False, ci=(0.5, 0.5))

    scorer = TCAVScorer(
        model_path=model_path,
        biomarker_name=biomarker,
        concept_csv=biomarker_csv,
        img_dir=img_dir,
        device=device,
        concept_dir=concept_dir,
        target_layer_name=target_layer_name,
        classifier_type=classifier_type,
        cav_accuracy_threshold=cav_accuracy_threshold,
        smoothgrad=smoothgrad,
        smoothgrad_samples=smoothgrad_samples,
        smoothgrad_noise=smoothgrad_noise,
    )

    df = pd.read_csv(biomarker_csv)
    df.columns = [c.strip() for c in df.columns]
    
    target_row = df[df["Image ID"] == test_image_id]
    if len(target_row) > 0:
        target_val = target_row[biomarker].values[0]
        all_ids = df.loc[df[biomarker] == target_val, "Image ID"].tolist()
    else:
        all_ids = df["Image ID"].tolist()

    test_ids = [test_image_id]
    for img_id in all_ids:
        if img_id != test_image_id:
            test_ids.append(img_id)
        if len(test_ids) >= 15:
            break

    try:
        mean_score, p_value, is_significant, details = \
            scorer.compute_tcav_with_statistical_testing(
                test_image_ids=test_ids,
                n_random_runs=n_random_runs,
            )
    except LowCAVAccuracyError as e:
        print(f"[TCAV] Aborted: {e}")
        return TCAVFloat(np.nan, p_value=1.0, is_significant=False, ci=(np.nan, np.nan),
                         details={"error": str(e), "mean_cav_accuracy": 0.0})

    print(f"\n{'='*50}")
    print(f"TCAV Report — {biomarker}")
    print(f"{'='*50}")
    print(f"  Mean TCAV score        : {mean_score:.4f}")
    print(f"  Mean CAV accuracy      : {details['mean_cav_accuracy']:.4f}")
    print(f"  Confidence Interval    : [{details['ci_low']:.4f}, {details['ci_high']:.4f}]")
    print(f"  p-value                : {p_value:.4f}")
    print(f"  Statistically signif.  : {is_significant}")
    print(f"  Relative sensitivity   : {details['relative_sensitivity']:.4f}")
    print(f"  Concept run scores     : {[f'{s:.3f}' for s in details['concept_scores']]}")
    print(f"  Random run scores      : {[f'{s:.3f}' for s in details['random_scores']]}")
    print(f"{'='*50}\n")

    return TCAVFloat(
        mean_score,
        p_value=p_value,
        is_significant=is_significant,
        ci=(details['ci_low'], details['ci_high']),
        details=details
    )
