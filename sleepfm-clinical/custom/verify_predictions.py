#!/usr/bin/env python3
"""
Verification script to evaluate sleep staging predictions.

This script calculates various metrics to assess the quality of predictions:
- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- Cohen's Kappa

It processes all prediction CSV files in /custom/predictions/ and generates
an evaluation report.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    classification_report
)
import sys

def load_predictions(predictions_dir):
    """
    Load all prediction CSV files.
    
    Args:
        predictions_dir: Directory containing prediction CSV files
        
    Returns:
        Tuple of (all_ground_truth, all_predictions)
    """
    all_ground_truth = []
    all_predictions = []
    
    prediction_files = sorted([
        f for f in os.listdir(predictions_dir)
        if f.endswith('_predictions.csv')
    ])
    
    print(f"Loading {len(prediction_files)} prediction files...")
    
    for pred_file in prediction_files:
        pred_path = os.path.join(predictions_dir, pred_file)
        df = pd.read_csv(pred_path)
        
        all_ground_truth.extend(df['GroundTruth'].tolist())
        all_predictions.extend(df['Predicted'].tolist())
    
    return np.array(all_ground_truth), np.array(all_predictions)


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    
    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted-averaged metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'cohen_kappa': kappa
    }
    
    return metrics


def print_metrics(metrics):
    """Print metrics in a readable format."""
    stage_names = ['Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'REM']
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    print("\n" + "-"*60)
    print("Macro-Averaged Metrics:")
    print("-"*60)
    print(f"Precision: {metrics['precision_macro']*100:.2f}%")
    print(f"Recall:    {metrics['recall_macro']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_macro']*100:.2f}%")
    
    print("\n" + "-"*60)
    print("Weighted-Averaged Metrics:")
    print("-"*60)
    print(f"Precision: {metrics['precision_weighted']*100:.2f}%")
    print(f"Recall:    {metrics['recall_weighted']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_weighted']*100:.2f}%")
    
    print("\n" + "-"*60)
    print("Per-Class Metrics:")
    print("-"*60)
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-"*60)
    
    for i, stage_name in enumerate(stage_names):
        precision = metrics['precision_per_class'][i] * 100
        recall = metrics['recall_per_class'][i] * 100
        f1 = metrics['f1_per_class'][i] * 100
        support = metrics['support_per_class'][i]
        
        print(f"{stage_name:<12} {precision:>10.2f}%  {recall:>10.2f}%  {f1:>10.2f}%  {support:>10.0f}")
    
    print("\n" + "-"*60)
    print("Confusion Matrix:")
    print("-"*60)
    print(f"{'True Pred':<12}", end="")
    for stage_name in stage_names:
        print(f"{stage_name:<12}", end="")
    print()
    print("-"*60)
    
    cm = metrics['confusion_matrix']
    for i, stage_name in enumerate(stage_names):
        print(f"{stage_name:<12}", end="")
        for j in range(len(stage_names)):
            print(f"{cm[i, j]:<12}", end="")
        print()
    
    print("="*60)


def save_metrics(metrics, output_path):
    """Save metrics to CSV file."""
    stage_names = ['Wake', 'Stage 1', 'Stage 2', 'Stage 3', 'REM']
    
    # Create per-class metrics dataframe
    per_class_data = {
        'Class': stage_names,
        'Precision': metrics['precision_per_class'],
        'Recall': metrics['recall_per_class'],
        'F1-Score': metrics['f1_per_class'],
        'Support': metrics['support_per_class']
    }
    per_class_df = pd.DataFrame(per_class_data)
    
    # Create overall metrics dataframe
    overall_data = {
        'Metric': [
            'Accuracy',
            'Cohen\'s Kappa',
            'Precision (Macro)',
            'Recall (Macro)',
            'F1-Score (Macro)',
            'Precision (Weighted)',
            'Recall (Weighted)',
            'F1-Score (Weighted)'
        ],
        'Value': [
            metrics['accuracy'],
            metrics['cohen_kappa'],
            metrics['precision_macro'],
            metrics['recall_macro'],
            metrics['f1_macro'],
            metrics['precision_weighted'],
            metrics['recall_weighted'],
            metrics['f1_weighted']
        ]
    }
    overall_df = pd.DataFrame(overall_data)
    
    # Save to CSV
    per_class_path = output_path.replace('.csv', '_per_class.csv')
    overall_path = output_path.replace('.csv', '_overall.csv')
    
    per_class_df.to_csv(per_class_path, index=False)
    overall_df.to_csv(overall_path, index=False)
    
    # Save confusion matrix
    cm_path = output_path.replace('.csv', '_confusion_matrix.csv')
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=stage_names,
        columns=stage_names
    )
    cm_df.to_csv(cm_path)
    
    print(f"\n✓ Metrics saved:")
    print(f"  - {os.path.basename(overall_path)}")
    print(f"  - {os.path.basename(per_class_path)}")
    print(f"  - {os.path.basename(cm_path)}")


def main():
    """Main function to evaluate predictions."""
    
    predictions_dir = "/home/nummm/Documents/CEPP/sleepfm-clinical/custom/predictions"
    output_path = os.path.join(predictions_dir, "evaluation_metrics.csv")
    
    print("\n" + "="*60)
    print("Sleep Staging Prediction Evaluation")
    print("="*60)
    print(f"Predictions directory: {predictions_dir}")
    print("="*60)
    
    # Check if predictions directory exists
    if not os.path.exists(predictions_dir):
        print(f"\n✗ Error: Predictions directory not found: {predictions_dir}")
        print("Please run predict_sleep_stages.py first")
        return 1
    
    # Check if there are any prediction files
    prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('_predictions.csv')]
    if not prediction_files:
        print(f"\n✗ Error: No prediction files found in {predictions_dir}")
        print("Please run predict_sleep_stages.py first")
        return 1
    
    # Load predictions
    y_true, y_pred = load_predictions(predictions_dir)
    
    print(f"Total epochs: {len(y_true)}")
    print(f"Total subjects: {len(prediction_files)}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save metrics
    save_metrics(metrics, output_path)
    
    print("\n✅ Evaluation complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
