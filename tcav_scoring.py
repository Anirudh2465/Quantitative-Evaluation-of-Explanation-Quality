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
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


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

    # Target layer — same bottleneck used for Grad-CAM everywhere else
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
    ):
        self.model_path = model_path
        self.biomarker_name = biomarker_name
        self.concept_csv = concept_csv
        self.img_dir = img_dir
        self.device = torch.device(device)
        self.activation_threshold = activation_threshold
        self.concept_dir = concept_dir

        # Load model
        self.model = _load_model(model_path, device=self.device)

        # Resolve target layer
        self.target_layer = self._resolve_layer(self.TARGET_LAYER_NAME)

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
        """Train a linear classifier separating concept from random activations.

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

        clf = SGDClassifier(
            loss="hinge",
            max_iter=1000,
            tol=1e-4,
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

    def _compute_directional_sensitivity(self, image_id, cav):
        """Compute the directional derivative of the model output w.r.t. *cav*.

        S(x) = ∇_{h_l} f(x)  ·  v_cav

        Uses self.grad_cache to cache gradient computations across multiple runs.
        """
        if image_id in self.grad_cache:
            grad_pooled = self.grad_cache[image_id]
        else:
            captured = {}

            def hook_fn(_module, _input, output):
                captured["act"] = output

            handle = self.target_layer.register_forward_hook(hook_fn)

            try:
                tensor = self._load_image(image_id).unsqueeze(0).to(self.device)
                tensor.requires_grad_(False)

                # Forward
                self.model.zero_grad()
                output = self.model(tensor)

                # The activation captured by the hook
                act = captured["act"]  # (1, C, H, W)

                # We need to compute grad of output w.r.t. the *raw* activation
                # then average-pool the gradient.
                grad = torch.autograd.grad(
                    outputs=output,
                    inputs=act,
                    retain_graph=True,
                    create_graph=False,
                )[0]  # (1, C, H, W)

                # Average-pool the gradient to match CAV shape
                grad_pooled = torch.nn.functional.adaptive_avg_pool2d(grad, 1)
                grad_pooled = grad_pooled.view(-1).cpu().numpy()  # (C,)
                
                self.grad_cache[image_id] = grad_pooled

            finally:
                handle.remove()

        # Directional derivative = dot(gradient, cav)
        sensitivity = float(np.dot(grad_pooled, cav))
        return sensitivity

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
                print("[TCAV] WARNING: Not enough concept images. Returning 0.5.")
                return 0.5, 0.0
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
                print("[TCAV] WARNING: Not enough random images. Returning 0.5.")
                return 0.5, 0.0
            random_acts = self._extract_activations(random_ids)

        if concept_acts.shape[0] == 0 or random_acts.shape[0] == 0:
            return 0.5, 0.0

        # Train CAV
        cav, accuracy = self._train_cav(concept_acts, random_acts)

        # Compute directional sensitivity for each test image
        positive_count = 0
        total_count = 0

        for img_id in test_image_ids:
            try:
                sens = self._compute_directional_sensitivity(img_id, cav)
                if sens > 0:
                    positive_count += 1
                total_count += 1
            except Exception as e:
                print(f"[TCAV] Sensitivity error for image {img_id}: {e}")
                continue

        if total_count == 0:
            return 0.5, accuracy

        tcav_score = positive_count / total_count
        return tcav_score, accuracy

    def compute_tcav_with_statistical_testing(
        self, test_image_ids, n_random_runs=10, significance_level=0.05
    ):
        """Run TCAV multiple times with shuffled random sets and test significance.

        Parameters
        ----------
        test_image_ids : list
            Image IDs to evaluate.
        n_random_runs : int
            Number of random baseline runs for the t-test.
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
                # We have folder concept activations!
                # For negative activations, if folder_neg_acts is available and large enough, sample from it.
                if folder_neg_acts is not None and folder_neg_acts.shape[0] >= 5:
                    idx = rng.choice(
                        folder_neg_acts.shape[0],
                        size=min(folder_neg_acts.shape[0], folder_pos_acts.shape[0]),
                        replace=False,
                    )
                    shuffled_neg_acts = folder_neg_acts[idx]
                else:
                    # Fallback: sample random activations from dataset
                    shuffled_neg_ids = rng.choice(
                        negative_ids,
                        size=min(len(negative_ids), folder_pos_acts.shape[0]),
                        replace=False,
                    ).tolist()
                    shuffled_neg_acts = self._extract_activations(shuffled_neg_ids)

                score_c, acc_c = self.compute_tcav_score(
                    test_image_ids,
                    concept_acts=folder_pos_acts,
                    random_acts=shuffled_neg_acts,
                )
            else:
                # Standard CSV flow
                shuffled_neg = rng.choice(
                    negative_ids,
                    size=min(len(negative_ids), len(concept_ids)),
                    replace=False,
                ).tolist()

                score_c, acc_c = self.compute_tcav_score(
                    test_image_ids,
                    concept_ids=concept_ids,
                    random_ids=shuffled_neg,
                )
            concept_scores.append(score_c)
            cav_accuracies.append(acc_c)

            # --- Random run: random "concept" vs random "negative" ---
            shuffled_all = rng.permutation(all_ids).tolist()
            mid = len(shuffled_all) // 2
            rand_concept = shuffled_all[:mid]
            rand_negative = shuffled_all[mid:]

            score_r, _ = self.compute_tcav_score(
                test_image_ids,
                concept_ids=rand_concept,
                random_ids=rand_negative,
            )
            random_scores.append(score_r)

        concept_scores = np.array(concept_scores)
        random_scores = np.array(random_scores)

        mean_tcav_score = float(np.mean(concept_scores))
        mean_cav_accuracy = float(np.mean(cav_accuracies))

        # One-sided t-test: concept scores > random scores
        t_stat = 0.0
        if len(concept_scores) > 1 and (np.std(concept_scores) > 0 or np.std(random_scores) > 0):
            t_stat, p_two = stats.ttest_ind(concept_scores, random_scores)
            # One-sided: concept > random
            p_value = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
        elif np.mean(concept_scores) > np.mean(random_scores):
            p_value = 0.0
            t_stat = float('inf')
        else:
            p_value = 1.0

        is_significant = p_value < significance_level

        details = {
            "concept_scores": concept_scores.tolist(),
            "random_scores": random_scores.tolist(),
            "cav_accuracies": cav_accuracies,
            "mean_cav_accuracy": mean_cav_accuracy,
            "t_stat": float(t_stat) if len(concept_scores) > 1 else 0.0,
        }

        return mean_tcav_score, p_value, is_significant, details


# ──────────────────────────────────────────────
# Convenience function (matches evaluate_cam_comprehensive interface)
# ──────────────────────────────────────────────

def evaluate_tcav(biomarker, image, img_dir, biomarker_csv, device="cpu",
                  n_random_runs=10, concept_dir=None):
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

    Returns
    -------
    tcav_score : float
        TCAV score in [0, 1].  Higher means the concept aligns with
        the model's decision for this image.
    """
    model_path = f"{biomarker}_model.pth"

    if not os.path.exists(model_path):
        # Fallback: try checkpoint path
        model_path = f"checkpoint_{biomarker}.pth"
        if not os.path.exists(model_path):
            print(f"[TCAV] Model not found for {biomarker}. Returning 0.5.")
            return 0.5

    # Extract image ID from path (e.g. "dataset_preprocessed/2.jpg" → 2)
    basename = os.path.splitext(os.path.basename(image))[0]
    try:
        test_image_id = int(basename)
    except ValueError:
        print(f"[TCAV] Cannot parse image ID from '{image}'. Returning 0.5.")
        return 0.5

    scorer = TCAVScorer(
        model_path=model_path,
        biomarker_name=biomarker,
        concept_csv=biomarker_csv,
        img_dir=img_dir,
        device=device,
        concept_dir=concept_dir,
    )

    # Get all image IDs from the same biomarker CSV to build a test set
    df = pd.read_csv(biomarker_csv)
    df.columns = [c.strip() for c in df.columns]
    
    # Get label of target image to filter neighbors of the same class
    target_row = df[df["Image ID"] == test_image_id]
    if len(target_row) > 0:
        target_val = target_row[biomarker].values[0]
        all_ids = df.loc[df[biomarker] == target_val, "Image ID"].tolist()
    else:
        all_ids = df["Image ID"].tolist()

    # Build a small test set: the target image + a few neighbours of the SAME class
    test_ids = [test_image_id]
    for img_id in all_ids:
        if img_id != test_image_id:
            test_ids.append(img_id)
        if len(test_ids) >= 15:
            break

    mean_score, p_value, is_significant, details = \
        scorer.compute_tcav_with_statistical_testing(
            test_image_ids=test_ids,
            n_random_runs=n_random_runs,
        )

    print(f"\n{'='*50}")
    print(f"TCAV Report — {biomarker}")
    print(f"{'='*50}")
    print(f"  Mean TCAV score        : {mean_score:.4f}")
    print(f"  Mean CAV accuracy      : {details['mean_cav_accuracy']:.4f}")
    print(f"  p-value                : {p_value:.4f}")
    print(f"  Statistically signif.  : {is_significant}")
    print(f"  Concept run scores     : {[f'{s:.3f}' for s in details['concept_scores']]}")
    print(f"  Random run scores      : {[f'{s:.3f}' for s in details['random_scores']]}")
    print(f"{'='*50}\n")

    # If concept is not significant, dampen the score toward 0.5
    if not is_significant:
        mean_score = 0.5 + (mean_score - 0.5) * 0.5
        print(f"[TCAV] Concept not significant — dampened score: {mean_score:.4f}")

    return mean_score
