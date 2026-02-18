#!/usr/bin/env python3
"""
Sleep Staging Evaluation Script for BENDR
Evaluates preprocessed EEG data using pre-trained BENDR model
Processes subjects sequentially for low RAM usage

Input: Preprocessed FIF files from custom/preprocessing_output/
Output: Predictions and metrics in custom/evaluation_results/
"""

# fix import error
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable

import os
import sys
import json
import torch
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

from dn3_ext import BENDRClassification
import torch.nn.functional as F

# Suppress MNE logging
mne.set_log_level('WARNING')

# Configuration
PREPROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'preprocessing_output'))
ENCODER_WEIGHTS = '/home/nummm/Documents/CEPP/BENDR/weights/encoder.pt'
CONTEXT_WEIGHTS = '/home/nummm/Documents/CEPP/BENDR/weights/contextualizer.pt'
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'evaluation_results'))

# Model parameters (matching BENDR pre-training)
NUM_CLASSES = 5
ENCODER_H = 512
CONTEXTUALIZER_HIDDEN = 3076
SFREQ = 256
TLEN = 30  # epoch length in seconds

# Sleep stage mapping
SLEEP_STAGES = {
    0: 'Wake',
    1: 'N1',
    2: 'N2',
    3: 'N3',
    4: 'REM'
}

STAGE_TO_INT = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,  # Map N4 to N3
    'Sleep stage R': 4
}


def load_bendr_model(encoder_path, context_path, num_classes=5, device='cpu'):
    """
    Load pre-trained BENDR model for inference
    
    Parameters:
    -----------
    encoder_path : str
        Path to encoder weights
    context_path : str
        Path to contextualizer weights
    num_classes : int
        Number of sleep stage classes
    device : str
        Device to load model on ('cpu' or 'cuda')
    
    Returns:
    --------
    model : BENDRClassification
        Loaded model in evaluation mode
    """
    print("Loading BENDR model...")
    
    # Check if weights exist
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder weights not found: {encoder_path}")
    if not os.path.exists(context_path):
        raise FileNotFoundError(f"Contextualizer weights not found: {context_path}")
    
    # Create model with dummy parameters (will be overridden by loaded weights)
    # We need to know the input shape - use typical sleep EEG setup
    samples = SFREQ * TLEN  # 256 Hz * 30 seconds = 7680 samples
    channels = 2  # Minimum channels (will work with more)
    
    model = BENDRClassification(
        targets=num_classes,
        samples=samples,
        channels=channels,
        encoder_h=ENCODER_H,
        contextualizer_hidden=CONTEXTUALIZER_HIDDEN,
        multi_gpu=False
    )
    
    # Load pre-trained weights
    print(f"  Loading encoder from: {encoder_path}")
    print(f"  Loading contextualizer from: {context_path}")
    
    model.load_pretrained_modules(
        encoder_path, 
        context_path,
        freeze_encoder=True,
        freeze_contextualizer=True,
        strict=False
    )
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully on {device}")
    return model


def load_subject_data(fif_path):
    """
    Load preprocessed FIF file and extract epochs with labels
    
    Parameters:
    -----------
    fif_path : str
        Path to preprocessed FIF file
    
    Returns:
    --------
    epochs_data : np.ndarray
        Epoch data (n_epochs, n_channels, n_samples)
    labels : np.ndarray
        Sleep stage labels (n_epochs,)
    """
    print(f"  Loading data from: {fif_path}")
    
    # Load raw data
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    
    # Get annotations
    annotations = raw.annotations
    
    if len(annotations) == 0:
        print("  Warning: No annotations found in file")
        return None, None
    
    # Create events from annotations
    events, event_id = mne.events_from_annotations(raw, event_id=STAGE_TO_INT, verbose=False)
    
    # Create epochs (30-second windows)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=TLEN - 1/SFREQ,  # 30 seconds minus one sample
        baseline=None,
        preload=True,
        verbose=False
    )
    
    # Get data and labels
    epochs_data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    labels = epochs.events[:, -1]  # Event IDs are the labels
    
    print(f"  Loaded {len(labels)} epochs with {epochs_data.shape[1]} channels")
    
    return epochs_data, labels


def predict_subject(model, data, device='cpu', batch_size=1):
    """
    Make predictions for one subject (sequential processing)
    
    Parameters:
    -----------
    model : BENDRClassification
        Loaded BENDR model
    data : np.ndarray
        Epoch data (n_epochs, n_channels, n_samples)
    device : str
        Device to run inference on
    batch_size : int
        Number of epochs to process at once (keep at 1 for low RAM)
    
    Returns:
    --------
    predictions : np.ndarray
        Predicted sleep stages (n_epochs,)
    probabilities : np.ndarray
        Class probabilities (n_epochs, n_classes)
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    n_epochs = data.shape[0]
    
    with torch.no_grad():
        for i in range(0, n_epochs, batch_size):
            # Get batch
            batch_data = data[i:i+batch_size]
            
            # Convert to tensor
            batch_tensor = torch.FloatTensor(batch_data).to(device)
            
            # Forward pass
            outputs = model(batch_tensor)
            
            # Get probabilities and predictions
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store results
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            
            # Clean up
            del batch_tensor, outputs, probs, preds
            
    return np.array(all_predictions), np.array(all_probabilities)


def evaluate_subject(predictions, true_labels, subject_id):
    """
    Calculate metrics for one subject
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted labels
    true_labels : np.ndarray
        True labels
    subject_id : str
        Subject identifier
    
    Returns:
    --------
    metrics : dict
        Dictionary of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, 
        predictions, 
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=list(range(NUM_CLASSES)))
    
    # Create metrics dictionary
    metrics = {
        'subject_id': subject_id,
        'n_epochs': len(true_labels),
        'overall_accuracy': float(accuracy),
        'per_class_metrics': {}
    }
    
    for i in range(NUM_CLASSES):
        stage_name = SLEEP_STAGES[i]
        metrics['per_class_metrics'][stage_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def save_predictions(predictions, probabilities, true_labels, subject_id, output_dir):
    """
    Save predictions to CSV file
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted labels
    probabilities : np.ndarray
        Class probabilities
    true_labels : np.ndarray
        True labels
    subject_id : str
        Subject identifier
    output_dir : str
        Output directory
    """
    # Create DataFrame
    df = pd.DataFrame({
        'epoch': range(len(predictions)),
        'true_stage': [SLEEP_STAGES[label] for label in true_labels],
        'predicted_stage': [SLEEP_STAGES[pred] for pred in predictions],
        'true_label': true_labels,
        'predicted_label': predictions,
        'correct': predictions == true_labels
    })
    
    # Add probability columns
    for i in range(NUM_CLASSES):
        df[f'prob_{SLEEP_STAGES[i]}'] = probabilities[:, i]
    
    # Save to CSV
    output_file = os.path.join(output_dir, f'{subject_id}_predictions.csv')
    df.to_csv(output_file, index=False)
    print(f"  Saved predictions to: {output_file}")


def save_metrics(metrics, output_dir):
    """
    Save metrics to JSON file
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary
    output_dir : str
        Output directory
    """
    subject_id = metrics['subject_id']
    output_file = os.path.join(output_dir, f'{subject_id}_metrics.json')
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  Saved metrics to: {output_file}")


def generate_summary_report(all_metrics, output_dir):
    """
    Generate summary report across all subjects
    
    Parameters:
    -----------
    all_metrics : list
        List of metrics dictionaries
    output_dir : str
        Output directory
    """
    print("\nGenerating summary report...")
    
    # Create summary DataFrame
    summary_data = []
    for metrics in all_metrics:
        row = {
            'subject_id': metrics['subject_id'],
            'n_epochs': metrics['n_epochs'],
            'accuracy': metrics['overall_accuracy']
        }
        
        # Add per-class F1 scores
        for stage_name, stage_metrics in metrics['per_class_metrics'].items():
            row[f'f1_{stage_name}'] = stage_metrics['f1_score']
            row[f'support_{stage_name}'] = stage_metrics['support']
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate overall statistics
    overall_stats = {
        'total_subjects': len(all_metrics),
        'total_epochs': summary_df['n_epochs'].sum(),
        'mean_accuracy': summary_df['accuracy'].mean(),
        'std_accuracy': summary_df['accuracy'].std(),
        'min_accuracy': summary_df['accuracy'].min(),
        'max_accuracy': summary_df['accuracy'].max()
    }
    
    # Save summary CSV
    summary_file = os.path.join(output_dir, 'summary_report.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"  Saved summary to: {summary_file}")
    
    # Save overall statistics
    stats_file = os.path.join(output_dir, 'overall_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    print(f"  Saved statistics to: {stats_file}")
    
    # Aggregate confusion matrix
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for metrics in all_metrics:
        total_cm += np.array(metrics['confusion_matrix'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        total_cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=[SLEEP_STAGES[i] for i in range(NUM_CLASSES)],
        yticklabels=[SLEEP_STAGES[i] for i in range(NUM_CLASSES)]
    )
    plt.title('Aggregated Confusion Matrix - All Subjects')
    plt.ylabel('True Stage')
    plt.xlabel('Predicted Stage')
    plt.tight_layout()
    
    cm_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_file, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix to: {cm_file}")
    
    return overall_stats


def main():
    """
    Main function to process all subjects sequentially
    """
    print("="*70)
    print("BENDR Sleep Staging Evaluation")
    print("="*70)
    print(f"Preprocessed data: {PREPROCESSED_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Encoder weights: {ENCODER_WEIGHTS}")
    print(f"Context weights: {CONTEXT_WEIGHTS}")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory created: {OUTPUT_DIR}")
    
    # Check if preprocessed data exists
    if not os.path.exists(PREPROCESSED_DIR):
        print(f"\nError: Preprocessed data directory not found: {PREPROCESSED_DIR}")
        print("Please run preprocess_rawEEG.py first")
        sys.exit(1)
    
    # Get list of preprocessed files
    fif_files = sorted([f for f in os.listdir(PREPROCESSED_DIR) if f.endswith('.fif')])
    
    if len(fif_files) == 0:
        print(f"\nError: No preprocessed FIF files found in {PREPROCESSED_DIR}")
        sys.exit(1)
    
    print(f"\nFound {len(fif_files)} preprocessed files")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model once
    try:
        model = load_bendr_model(ENCODER_WEIGHTS, CONTEXT_WEIGHTS, NUM_CLASSES, device)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nPlease ensure pre-trained weights are downloaded and paths are correct:")
        print(f"  Encoder: {ENCODER_WEIGHTS}")
        print(f"  Contextualizer: {CONTEXT_WEIGHTS}")
        sys.exit(1)
    
    # Process each subject sequentially
    all_metrics = []
    success_count = 0
    error_count = 0
    
    for i, fif_file in enumerate(fif_files, 1):
        print(f"\n{'='*70}")
        print(f"Progress: {i}/{len(fif_files)}")
        print(f"{'='*70}")
        
        subject_id = fif_file.replace('_preprocessed.fif', '')
        fif_path = os.path.join(PREPROCESSED_DIR, fif_file)
        
        print(f"Processing subject: {subject_id}")
        
        try:
            # Load data
            data, labels = load_subject_data(fif_path)
            
            if data is None or labels is None:
                print(f"  Skipping {subject_id}: No valid data")
                error_count += 1
                continue
            
            # Make predictions
            print(f"  Making predictions...")
            predictions, probabilities = predict_subject(model, data, device, batch_size=1)
            
            # Evaluate
            print(f"  Calculating metrics...")
            metrics = evaluate_subject(predictions, labels, subject_id)
            
            # Save results
            save_predictions(predictions, probabilities, labels, subject_id, OUTPUT_DIR)
            save_metrics(metrics, OUTPUT_DIR)
            
            # Store metrics
            all_metrics.append(metrics)
            
            # Print summary
            print(f"  ✓ Accuracy: {metrics['overall_accuracy']:.4f} ({len(labels)} epochs)")
            
            success_count += 1
            
            # Clean up memory
            del data, labels, predictions, probabilities
            if device == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ Error processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    # Generate summary report
    if len(all_metrics) > 0:
        overall_stats = generate_summary_report(all_metrics, OUTPUT_DIR)
        
        # Print final summary
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Total files: {len(fif_files)}")
        print(f"Successfully processed: {success_count}")
        print(f"Errors: {error_count}")
        print(f"\nOverall Statistics:")
        print(f"  Total epochs: {overall_stats['total_epochs']}")
        print(f"  Mean accuracy: {overall_stats['mean_accuracy']:.4f}")
        print(f"  Std accuracy: {overall_stats['std_accuracy']:.4f}")
        print(f"  Min accuracy: {overall_stats['min_accuracy']:.4f}")
        print(f"  Max accuracy: {overall_stats['max_accuracy']:.4f}")
        print("="*70)
        print(f"\nResults saved to: {OUTPUT_DIR}")
    else:
        print("\n" + "="*70)
        print("No subjects were successfully processed!")
        print("="*70)


if __name__ == '__main__':
    main()
