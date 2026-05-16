import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


class UltrasoundDataset(Dataset):
    def __init__(self, csv_file, img_dir, biomarker_col, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.labels_df.columns = [col.strip() for col in self.labels_df.columns]
        self.img_dir = img_dir
        self.transform = transform
        self.biomarker_col = biomarker_col
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        img_name = f"{self.labels_df.iloc[idx, 0]}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image {img_path} not found. Using random image.")
            image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        
        label = self.labels_df.loc[idx, self.biomarker_col]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.FloatTensor([label])


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss, model, path='checkpoint.pt'):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=True):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        
    def __call__(self, metric):
        if self.mode == 'min':
            is_better = metric < self.best - 1e-4
        else:
            is_better = metric > self.best + 1e-4
            
        if is_better:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = old_lr * self.factor
                    param_group['lr'] = new_lr
                    if self.verbose:
                        print(f'Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}')
                self.counter = 0


def create_model(num_classes=1):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    for param in list(model.parameters())[:-40]:
        param.requires_grad = False
        
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.Sigmoid()
    )
    
    return model


def train_biomarker_classifier(biomarker_name, biomarker_csv, img_dir, 
                               batch_size=32, num_epochs=50, 
                               learning_rate=1e-4, patience=10,
                               output_model_path=None):
    """
    Train a binary classifier for a specific biomarker.
    
    Args:
        biomarker_name (str): Name of the biomarker (e.g., 'Neutral', 'Zoom')
        biomarker_csv (str): Path to CSV file with columns ['Image ID', biomarker_name]
        img_dir (str): Directory containing the images
        batch_size (int): Batch size for training
        num_epochs (int): Maximum number of epochs
        learning_rate (float): Initial learning rate
        patience (int): Early stopping patience
        output_model_path (str): Path to save the trained model (default: {biomarker_name}.pth)
    
    Returns:
        dict: Training results including model, history, metrics
    """
    
    print(f"\n{'='*60}")
    print(f"Training classifier for biomarker: {biomarker_name}")
    print(f"{'='*60}\n")
    
    # Set default output path
    if output_model_path is None:
        output_model_path = f"{biomarker_name}.pth"
    
    # Fix for Windows multiprocessing
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv(biomarker_csv)
    df.columns = [col.strip() for col in df.columns]
    
    # Check for missing values
    if df[biomarker_name].isna().any():
        print(f"Warning: Found {df[biomarker_name].isna().sum()} missing values. Filling with 0.")
        df[biomarker_name] = df[biomarker_name].fillna(0)
    
    # Check class distribution
    class_counts = df[biomarker_name].value_counts()
    print(f"Class distribution: {dict(class_counts)}")
    
    # Split data
    try:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[biomarker_name], random_state=42)
        print("Using stratified split")
    except ValueError as e:
        print(f"Warning: Stratification failed - {e}. Using random split.")
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save splits
    train_csv = f'train_{biomarker_name}.csv'
    val_csv = f'val_{biomarker_name}.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # Create datasets
    train_dataset = UltrasoundDataset(train_csv, img_dir, biomarker_name, train_transform)
    val_dataset = UltrasoundDataset(val_csv, img_dir, biomarker_name, val_transform)
    
    # Handle class imbalance with weighted sampler
    try:
        class_counts = train_df[biomarker_name].value_counts().to_dict()
        num_samples = len(train_df)
        class_weights = {class_val: num_samples / (len(class_counts) * count) 
                        for class_val, count in class_counts.items()}
        sample_weights = [class_weights[float(label)] for label in train_df[biomarker_name]]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    except KeyError as e:
        print(f"Warning: Issue with class weights: {e}. Using default sampler.")
        sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Callbacks
    checkpoint_path = f'checkpoint_{biomarker_name}.pth'
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_auc': []
    }
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.numel()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_preds / total_preds
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct_preds = 0
        val_total_preds = 0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                val_correct_preds += (predicted == labels).sum().item()
                val_total_preds += labels.numel()
                
                all_targets.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct_preds / val_total_preds
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Calculate AUC
        all_targets = np.array(all_targets)
        all_outputs = np.array(all_outputs)
        try:
            auc = roc_auc_score(all_targets, all_outputs)
            history['val_auc'].append(auc)
        except ValueError as e:
            print(f"Warning: AUC calculation error - {e}")
            auc = 0
            history['val_auc'].append(0)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, AUC: {auc:.4f}")
        
        # Apply callbacks
        lr_scheduler(val_loss)
        early_stopping(val_loss, model, path=checkpoint_path)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    # Final evaluation
    print("\nFinal Evaluation:")
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(predicted)
            all_targets.extend(labels.numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Classification report
    target_names = [f'No {biomarker_name}', biomarker_name]
    report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0)
    
    # Calculate metrics
    try:
        final_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        final_auc = 0
    
    confusion = {
        'true_positives': np.sum((all_targets == 1) & (all_preds == 1)),
        'false_positives': np.sum((all_targets == 0) & (all_preds == 1)),
        'true_negatives': np.sum((all_targets == 0) & (all_preds == 0)),
        'false_negatives': np.sum((all_targets == 1) & (all_preds == 0))
    }
    
    sensitivity = confusion['true_positives'] / (confusion['true_positives'] + confusion['false_negatives'] + 1e-10)
    specificity = confusion['true_negatives'] / (confusion['true_negatives'] + confusion['false_positives'] + 1e-10)
    
    print("\nClassification Report:")
    print(report)
    print(f"\nAUC: {final_auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Confusion Matrix: {confusion}")
    
    # Save final model
    torch.save(model.state_dict(), output_model_path)
    print(f"\nModel saved to: {output_model_path}")
    
    # Clean up temporary files
    for f in [train_csv, val_csv, checkpoint_path]:
        if os.path.exists(f):
            os.remove(f)
    
    # Return results
    return {
        'model': model,
        'history': history,
        'metrics': {
            'auc': final_auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': confusion
        },
        'classification_report': report,
        'model_path': output_model_path
    }


def predict_image(image_path, model_path, biomarker_name, transform=None):
    """
    Make predictions on a single image.
    
    Args:
        image_path (str): Path to the image
        model_path (str): Path to the trained model
        biomarker_name (str): Name of the biomarker
        transform: Optional transform (uses default if None)
    
    Returns:
        dict: Prediction results
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_classes=1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probability = output.cpu().numpy()[0][0]
        prediction = bool(probability > 0.5)
    
    return {
        biomarker_name: prediction,
        'probability': float(probability),
        'confidence': float(abs(probability - 0.5) + 0.5)
    }