# Helper Train Module

## Overview

`helper_train.py` provides training utilities for the mulEEG sleep stage classification framework. It implements a **two-stage training approach**:

1. **Contrastive Pretraining** (`sleep_pretrain`) - Self-supervised learning using weak and strong augmentations
2. **Fine-tuning** (`sleep_ft`) - Supervised classification with pretrained encoder

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    sleep_pretrain                        │
│  ┌────────────────────────────────────────────────┐     │
│  │  Contrastive Learning (Self-Supervised)        │     │
│  │  - Time domain loss                            │     │
│  │  - Spectral domain loss                        │     │
│  │  - Fusion loss                                 │     │
│  │  - Intra-batch contrastive loss                │     │
│  └────────────────────────────────────────────────┘     │
│                          ↓                               │
│              Save EEG Encoder Weights                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                      sleep_ft                            │
│  ┌────────────────────────────────────────────────┐     │
│  │  Fine-tuning (Supervised Classification)       │     │
│  │  - Load pretrained encoder                     │     │
│  │  - Train on labeled sleep stage data           │     │
│  │  - 5-class classification (W, N1, N2, N3, REM) │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Classes

### 1. `sleep_pretrain`

Implements contrastive pretraining using weak and strong augmentations.

#### Initialization

```python
sleep_pretrain(config, name, dataloader, wandb_logger)
```

**Parameters:**
- `config` (Config): Configuration object containing hyperparameters
- `name` (str): Experiment name for saving checkpoints
- `dataloader` (DataLoader): DataLoader providing weak/strong augmentation pairs
- `wandb_logger`: Weights & Biases logger for experiment tracking

#### Key Methods

##### `training_step(batch, batch_idx)`

Performs one training step with contrastive learning.

**Input:**
- `batch`: Tuple of `(weak_aug, strong_aug)` tensors

**Returns:**
- `loss`: Combined contrastive loss

##### `training_epoch_end(outputs)`

Aggregates and logs epoch-level metrics.

**Logged Metrics:**
- Epoch Loss (total)
- Fusion Loss
- Time Loss
- Spectral Loss
- Intra-batch Loss
- Learning Rate

##### `on_epoch_end()`

Saves checkpoints:
- `{name}.pt` - EEG encoder only (for fine-tuning)
- `{name}_full.pt` - Full model state (for resuming pretraining)

##### `do_kfold()`

Performs 5-fold cross-validation evaluation during pretraining.

**Returns:**
- Average F1, Mean F1, Kappa, Balanced Accuracy, Accuracy across folds

##### `fit()`

Main pretraining loop.

**Process:**
1. Train for `num_epoch` epochs
2. Every 4 epochs (after epoch 80): Run 5-fold evaluation
3. Save best model based on F1 score
4. Log all metrics to WandB

**Saved Checkpoints:**
- `{name}_best.pt` - Best model by F1 score
- `{name}_mean_best.pt` - Best model by mean F1 score

---

### 2. `sleep_ft`

Implements supervised fine-tuning for sleep stage classification.

#### Initialization

```python
sleep_ft(chkpoint_pth, config, train_dl, valid_dl, pret_epoch, wandb_logger)
```

**Parameters:**
- `chkpoint_pth` (str): Path to pretrained encoder checkpoint
- `config` (Config): Configuration object
- `train_dl` (DataLoader): Training data loader
- `valid_dl` (DataLoader): Validation data loader
- `pret_epoch` (int): Pretraining epoch number (for logging)
- `wandb_logger`: WandB logger

#### Key Methods

##### `training_step(batch, batch_idx)`

Performs one supervised training step.

**Input:**
- `batch`: Tuple of `(data, labels)`

**Returns:**
- `loss`: CrossEntropyLoss

##### `validation_step(batch, batch_idx)`

Performs one validation step.

**Returns:**
- Dictionary with `loss`, `acc`, `preds`, `target`

##### `validation_epoch_end(outputs)`

Computes epoch-level validation metrics.

**Computed Metrics:**
- Accuracy
- Macro F1 Score
- Cohen's Kappa
- Balanced Accuracy
- Confusion Matrix (logged to WandB)

##### `fit()`

Main fine-tuning loop.

**Process:**
1. Train for `num_ft_epoch` epochs
2. Validate after each epoch
3. Track best metrics (F1, Kappa, Balanced Accuracy, Accuracy)
4. Print progress

**Returns:**
- `max_f1`: Best F1 score
- `mean_f1`: Average F1 across all epochs
- `max_kappa`: Best Cohen's Kappa
- `max_bal_acc`: Best balanced accuracy
- `max_acc`: Best accuracy

---

## Training Pipeline

### Complete Training Example

```python
import torch
from config import Config
from helper_train import sleep_pretrain, sleep_ft
from utils.dataloader import get_pretrain_dataloader, cross_data_generator
import wandb

# 1. Initialize configuration
config = Config()
wandb_logger = wandb.init(project="mulEEG", name="experiment_1")

# 2. Pretrain with contrastive learning
pretrain_loader = get_pretrain_dataloader(config)
pretrain_model = sleep_pretrain(
    config=config,
    name="pretrain_exp1",
    dataloader=pretrain_loader,
    wandb_logger=wandb_logger
)
pretrain_model.fit()

# 3. Fine-tune on labeled data (5-fold CV)
from sklearn.model_selection import KFold
import numpy as np

n_subjects = 100  # Total number of subjects
kfold = KFold(n_splits=5, shuffle=False)
idxs = np.arange(n_subjects)

for split, (train_idx, val_idx) in enumerate(kfold.split(idxs)):
    print(f"Fold {split + 1}/5")
    
    # Create data loaders for this fold
    train_dl, valid_dl = cross_data_generator(
        config.src_path, train_idx, val_idx, config
    )
    
    # Fine-tune
    ft_model = sleep_ft(
        chkpoint_pth=f"{config.exp_path}/pretrain_exp1_best.pt",
        config=config,
        train_dl=train_dl,
        valid_dl=valid_dl,
        pret_epoch=0,
        wandb_logger=wandb_logger
    )
    
    f1, mean_f1, kappa, bal_acc, acc = ft_model.fit()
    print(f"Fold {split + 1} - F1: {f1:.4f}, Kappa: {kappa:.4f}, Acc: {acc:.4f}")
```

---

## Loss Functions

### Pretraining Losses

The `contrast_loss` model computes multiple contrastive losses:

1. **Time Loss** - Contrastive loss in time domain
2. **Spectral Loss** - Contrastive loss in frequency domain
3. **Fusion Loss** - Combined time-frequency representation loss
4. **Intra Loss** - Intra-batch contrastive loss

**Total Loss:**
```
loss = time_loss + spectral_loss + fusion_loss + intra_loss
```

### Fine-tuning Loss

Standard **CrossEntropyLoss** for 5-class classification.

---

## Metrics

### Classification Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Accuracy** | Overall correct predictions | General performance |
| **Macro F1** | Average F1 across all classes | Balanced class performance |
| **Mean F1** | Average F1 across epochs | Training stability |
| **Cohen's Kappa** | Agreement beyond chance | Inter-rater reliability |
| **Balanced Accuracy** | Average recall per class | Imbalanced datasets |

### Sleep Stage Classes

| Label | Sleep Stage |
|-------|-------------|
| 0 | Wake |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 |
| 4 | REM |

---

## Checkpoints

### Pretraining Checkpoints

| Filename | Contents | Purpose |
|----------|----------|---------|
| `{name}.pt` | EEG encoder only | Fine-tuning initialization |
| `{name}_full.pt` | Full model + optimizer + epoch | Resume pretraining |
| `{name}_best.pt` | Best encoder by F1 | Best model for deployment |
| `{name}_mean_best.pt` | Best encoder by mean F1 | Alternative best model |

### Checkpoint Structure

```python
# Encoder-only checkpoint
{
    'eeg_model_state_dict': state_dict,
    'best_pretrain_epoch': int  # (for best checkpoints)
}

# Full checkpoint
{
    'model_state_dict': state_dict,
    'epoch': int
}
```

---

## Configuration

Required configuration parameters (from `config.py`):

```python
class Config:
    # Training
    num_epoch = 200          # Pretraining epochs
    num_ft_epoch = 50        # Fine-tuning epochs
    batch_size = 32
    lr = 0.001              # Learning rate
    beta1 = 0.9             # Adam beta1
    beta2 = 0.999           # Adam beta2
    
    # Paths
    src_path = "./data"     # Source data directory
    exp_path = "./checkpoints"  # Experiment output directory
```

---

## Optimizer & Scheduler

- **Optimizer:** Adam with weight decay (3e-5)
- **Scheduler:** ReduceLROnPlateau
  - Mode: minimize loss
  - Patience: 5 epochs
  - Factor: 0.2 (reduce LR by 80%)

---

## Evaluation Strategy

### During Pretraining

- Every 4 epochs (after epoch 80):
  - Run 5-fold cross-validation
  - Evaluate fine-tuned performance
  - Save best models

### During Fine-tuning

- After each epoch:
  - Compute validation metrics
  - Track best F1, Kappa, Balanced Accuracy
  - Log confusion matrix for best model

---

## Dependencies

```python
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import accuracy, f1, cohen_kappa
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import KFold
```

---

## Related Files

- [`config.py`](file:///home/nummm/Documents/CEPP/mulEEG/config.py) - Configuration settings
- [`models/model.py`](file:///home/nummm/Documents/CEPP/mulEEG/models/model.py) - Model architectures (`contrast_loss`, `ft_loss`)
- [`utils/dataloader.py`](file:///home/nummm/Documents/CEPP/mulEEG/utils/dataloader.py) - Data loading utilities
- [`preprocessing/custom/preprocess_custom.py`](file:///home/nummm/Documents/CEPP/mulEEG/preprocessing/custom/preprocess_custom.py) - Data preprocessing

---

## Notes

- **GPU Support:** Automatically uses CUDA if available
- **Logging:** All metrics logged to Weights & Biases
- **Checkpointing:** Models saved at regular intervals and when best performance achieved
- **Cross-Validation:** 5-fold CV used for robust evaluation
- **Two-Stage Training:** Pretraining provides better initialization for fine-tuning


## Key Functions

### 1. `load_data(data_dir, fold_idx)`

Loads preprocessed sleep stage data for a specific fold.

**Parameters:**
- `data_dir` (str): Path to directory containing preprocessed `.npz` files
- `fold_idx` (int): Fold index for cross-validation (0-based)

**Returns:**
- `X_train` (np.ndarray): Training data, shape `(n_samples, n_channels, sequence_length)`
- `y_train` (np.ndarray): Training labels, shape `(n_samples,)`
- `X_val` (np.ndarray): Validation data
- `y_val` (np.ndarray): Validation labels
- `X_test` (np.ndarray): Test data
- `y_test` (np.ndarray): Test labels

**Example:**
```python
X_train, y_train, X_val, y_val, X_test, y_test = load_data('./data', fold_idx=0)
```

---

### 2. `create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32)`

Creates PyTorch DataLoader objects for training and validation.

**Parameters:**
- `X_train`, `y_train`: Training data and labels
- `X_val`, `y_val`: Validation data and labels
- `batch_size` (int): Batch size for training (default: 32)

**Returns:**
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader): Validation data loader

**Example:**
```python
train_loader, val_loader = create_dataloaders(
    X_train, y_train, X_val, y_val, batch_size=64
)
```

---

### 3. `train_epoch(model, train_loader, criterion, optimizer, device)`

Trains the model for one epoch.

**Parameters:**
- `model` (nn.Module): PyTorch model to train
- `train_loader` (DataLoader): Training data loader
- `criterion`: Loss function (e.g., CrossEntropyLoss)
- `optimizer`: Optimizer (e.g., Adam, SGD)
- `device` (torch.device): Device to train on (CPU/GPU)

**Returns:**
- `avg_loss` (float): Average training loss for the epoch
- `accuracy` (float): Training accuracy (0-100)

**Example:**
```python
loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
print(f"Train Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
```

---

### 4. `validate(model, val_loader, criterion, device)`

Evaluates the model on validation data.

**Parameters:**
- `model` (nn.Module): PyTorch model to evaluate
- `val_loader` (DataLoader): Validation data loader
- `criterion`: Loss function
- `device` (torch.device): Device to evaluate on

**Returns:**
- `avg_loss` (float): Average validation loss
- `accuracy` (float): Validation accuracy (0-100)
- `predictions` (np.ndarray): Model predictions
- `targets` (np.ndarray): Ground truth labels

**Example:**
```python
val_loss, val_acc, preds, targets = validate(model, val_loader, criterion, device)
print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
```

---

### 5. `compute_metrics(y_true, y_pred, num_classes=5)`

Computes comprehensive classification metrics.

**Parameters:**
- `y_true` (np.ndarray): Ground truth labels
- `y_pred` (np.ndarray): Predicted labels
- `num_classes` (int): Number of classes (default: 5 for sleep stages)

**Returns:**
- `metrics` (dict): Dictionary containing:
  - `accuracy`: Overall accuracy
  - `precision`: Per-class precision
  - `recall`: Per-class recall
  - `f1_score`: Per-class F1 score
  - `confusion_matrix`: Confusion matrix
  - `macro_f1`: Macro-averaged F1 score
  - `weighted_f1`: Weighted F1 score

**Example:**
```python
metrics = compute_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

---

### 6. `save_checkpoint(model, optimizer, epoch, metrics, filepath)`

Saves model checkpoint with training state.

**Parameters:**
- `model` (nn.Module): Model to save
- `optimizer`: Optimizer state
- `epoch` (int): Current epoch number
- `metrics` (dict): Training/validation metrics
- `filepath` (str): Path to save checkpoint

**Example:**
```python
save_checkpoint(
    model, optimizer, epoch=10, 
    metrics={'val_acc': 0.85, 'val_loss': 0.42},
    filepath='./checkpoints/model_epoch10.pth'
)
```

---

### 7. `load_checkpoint(filepath, model, optimizer=None)`

Loads model checkpoint.

**Parameters:**
- `filepath` (str): Path to checkpoint file
- `model` (nn.Module): Model to load weights into
- `optimizer` (optional): Optimizer to load state into

**Returns:**
- `epoch` (int): Epoch number from checkpoint
- `metrics` (dict): Saved metrics

**Example:**
```python
epoch, metrics = load_checkpoint('./checkpoints/best_model.pth', model, optimizer)
print(f"Resumed from epoch {epoch}")
```

---

### 8. `plot_training_history(history, save_path=None)`

Plots training and validation metrics over epochs.

**Parameters:**
- `history` (dict): Dictionary with keys `train_loss`, `train_acc`, `val_loss`, `val_acc`
- `save_path` (str, optional): Path to save plot image

**Example:**
```python
history = {
    'train_loss': [0.8, 0.6, 0.5],
    'train_acc': [60, 70, 75],
    'val_loss': [0.7, 0.55, 0.48],
    'val_acc': [65, 72, 77]
}
plot_training_history(history, save_path='./plots/training.png')
```

---

### 9. `log_metrics(metrics, epoch, logger=None)`

Logs metrics to console and optionally to a logger (e.g., WandB, TensorBoard).

**Parameters:**
- `metrics` (dict): Metrics to log
- `epoch` (int): Current epoch
- `logger` (optional): External logger object

**Example:**
```python
log_metrics({'loss': 0.42, 'acc': 85.3}, epoch=10)
```

---

## Typical Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from helper_train import *

# 1. Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_data('./data', fold_idx=0)

# 2. Create dataloaders
train_loader, val_loader = create_dataloaders(
    X_train, y_train, X_val, y_val, batch_size=64
)

# 3. Initialize model, loss, optimizer
model = YourModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc, preds, targets = validate(model, val_loader, criterion, device)
    
    # Log
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(model, optimizer, epoch, 
                       {'val_acc': val_acc, 'val_loss': val_loss},
                       './checkpoints/best_model.pth')

# 5. Compute final metrics
metrics = compute_metrics(targets, preds)
print(f"Final Metrics: {metrics}")

# 6. Plot training history
plot_training_history(history, save_path='./plots/training.png')
```

## Sleep Stage Labels

The module expects sleep stage labels in the following format:

| Label | Sleep Stage |
|-------|-------------|
| 0 | Wake |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 |
| 4 | REM |

## Dependencies

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
```

## Notes

- All data should be preprocessed using the preprocessing scripts before using these utilities
- The module assumes data is in `.npz` format with keys `x`, `y`, `fs`
- GPU training is supported via the `device` parameter
- Checkpoints include model state, optimizer state, epoch number, and metrics

## Related Files

- [`preprocessing/custom/preprocess_custom.py`](file:///home/nummm/Documents/CEPP/mulEEG/preprocessing/custom/preprocess_custom.py) - Data preprocessing
- Model definition files (e.g., `models/mulEEG.py`)
- Training scripts that use these utilities
