"""
Evaluation script for mulEEG sleep stage classification model.

This script loads a pretrained model and evaluates it on preprocessed patient data.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Import model and config
from models.model import ft_loss
from config import Config


class PatientEvaluator:
    """Evaluates pretrained model on patient data."""
    
    def __init__(self, checkpoint_path: str, output_dir: str, device: str = "cpu"):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to pretrained model checkpoint
            output_dir: Directory containing patient folders with .npz files
            device: Device to run inference on (default: "cpu")
        """
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        
        # Initialize config
        self.config = Config()
        self.config.device = device
        
        # Load model
        print(f"Loading model from: {checkpoint_path}")
        self.model = ft_loss(checkpoint_path, self.config, device)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"Model loaded successfully on {device}")
        
    def load_patient_data(self, patient_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load patient data from .npz file.
        
        Args:
            patient_path: Path to patient folder
            
        Returns:
            Tuple of (data, labels)
        """
        patient_id = patient_path.name
        npz_file = patient_path / f"{patient_id}.npz"
        
        if not npz_file.exists():
            raise FileNotFoundError(f"File not found: {npz_file}")
        
        data = np.load(npz_file)
        x = data['x']  # Shape: (n_epochs, sequence_length)
        y = data['y']  # Shape: (n_epochs,)
        
        return x, y
    
    def predict_patient(self, x: np.ndarray, batch_size: int = 32, debug: bool = True) -> np.ndarray:
        """
        Run inference on patient data.
        
        Args:
            x: Input data, shape (n_epochs, sequence_length)
            batch_size: Batch size for inference
            debug: Whether to print debugging information
            
        Returns:
            Predictions, shape (n_epochs,)
        """
        # Add channel dimension: (n_epochs, 1, sequence_length)
        x_tensor = torch.from_numpy(x).unsqueeze(1).float()
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(x_tensor), batch_size):
                batch = x_tensor[i:i+batch_size].to(self.device)
                outputs = self.model(batch)
                
                # Debug: Print detailed information for first batch
                if debug and i == 0:
                    print(f"\n  🔍 Debug Information (First Batch):")
                    print(f"  {'='*60}")
                    print(f"  Input shape: {batch.shape}")
                    print(f"  Output logits shape: {outputs.shape}")
                    
                    # Show sample logits
                    print(f"\n  Sample Logits (first 3 samples):")
                    for j in range(min(3, len(outputs))):
                        logit = outputs[j].cpu().numpy()
                        pred = logit.argmax()
                        # Compute softmax probabilities
                        probs = torch.softmax(outputs[j], dim=0).cpu().numpy()
                        print(f"    Sample {j}:")
                        print(f"      Logits: {logit}")
                        print(f"      Probs:  {probs}")
                        print(f"      Pred:   {pred} (class with highest prob: {probs[pred]:.4f})")
                    
                    # Statistics across batch
                    print(f"\n  Batch Statistics:")
                    print(f"    Logits mean per class: {outputs.mean(dim=0).cpu().numpy()}")
                    print(f"    Logits std per class:  {outputs.std(dim=0).cpu().numpy()}")
                    
                    # Prediction distribution in first batch
                    batch_preds = outputs.argmax(dim=1).cpu().numpy()
                    unique, counts = np.unique(batch_preds, return_counts=True)
                    print(f"\n  Prediction distribution in first batch:")
                    for cls, cnt in zip(unique, counts):
                        print(f"    Class {cls}: {cnt}/{len(batch_preds)} ({100*cnt/len(batch_preds):.1f}%)")
                    print(f"  {'='*60}\n")
                
                preds = outputs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)
        
        return np.array(predictions)
    
    def evaluate_patient(self, patient_path: Path) -> Dict:
        """
        Evaluate model on a single patient.
        
        Args:
            patient_path: Path to patient folder
            
        Returns:
            Dictionary with evaluation metrics
        """
        patient_id = patient_path.name
        
        try:
            # Load data
            x, y_true = self.load_patient_data(patient_path)
            
            # Run inference
            y_pred = self.predict_patient(x)
            
            # Debug: Check unique labels
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            print(f"  Unique labels in y_true: {unique_true}")
            print(f"  Unique labels in y_pred: {unique_pred}")
            
            # Define all possible labels (0-4 for 5 sleep stages)
            all_labels = [0, 1, 2, 3, 4]
            
            # Compute metrics with explicit labels parameter
            metrics = {
                'patient_id': patient_id,
                'n_epochs': len(y_true),
                'accuracy': accuracy_score(y_true, y_pred),
                'macro_f1': f1_score(y_true, y_pred, average='macro', labels=all_labels, zero_division=0),
                'weighted_f1': f1_score(y_true, y_pred, average='weighted', labels=all_labels, zero_division=0),
                'kappa': cohen_kappa_score(y_true, y_pred, labels=all_labels),
                'balanced_acc': balanced_accuracy_score(y_true, y_pred),
            }
            
            # Per-class F1 scores with explicit labels
            per_class_f1 = f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
            metrics['f1_wake'] = per_class_f1[0]
            metrics['f1_n1'] = per_class_f1[1]
            metrics['f1_n2'] = per_class_f1[2]
            metrics['f1_n3'] = per_class_f1[3]
            metrics['f1_rem'] = per_class_f1[4]
            
            return metrics, y_true, y_pred
            
        except Exception as e:
            print(f"Error evaluating {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def evaluate_all_patients(self) -> pd.DataFrame:
        """
        Evaluate model on all patients in output directory.
        
        Returns:
            DataFrame with per-patient results
        """
        patient_folders = sorted([p for p in self.output_dir.iterdir() if p.is_dir()])
        
        print(f"\nFound {len(patient_folders)} patient folders")
        print("=" * 70)
        
        results = []
        all_y_true = []
        all_y_pred = []
        
        for i, patient_path in enumerate(patient_folders, 1):
            print(f"\nProcessing {i}/{len(patient_folders)}: {patient_path.name}")
            
            metrics, y_true, y_pred = self.evaluate_patient(patient_path)
            
            if metrics is not None:
                results.append(metrics)
                all_y_true.extend(y_true)
                all_y_pred.extend(y_pred)
                
                print(f"  Epochs: {metrics['n_epochs']}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Macro F1: {metrics['macro_f1']:.4f}")
                print(f"  Kappa: {metrics['kappa']:.4f}")
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Define all possible labels
        all_labels = [0, 1, 2, 3, 4]
        
        # Compute overall metrics with explicit labels
        print("\n" + "=" * 70)
        print("OVERALL EVALUATION RESULTS")
        print("=" * 70)
        print(f"Total Patients: {len(results)}")
        print(f"Total Epochs: {len(all_y_true)}")
        print(f"Overall Accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}")
        print(f"Overall Macro F1: {f1_score(all_y_true, all_y_pred, average='macro', labels=all_labels, zero_division=0):.4f}")
        print(f"Overall Weighted F1: {f1_score(all_y_true, all_y_pred, average='weighted', labels=all_labels, zero_division=0):.4f}")
        print(f"Overall Kappa: {cohen_kappa_score(all_y_true, all_y_pred, labels=all_labels):.4f}")
        print(f"Overall Balanced Acc: {balanced_accuracy_score(all_y_true, all_y_pred):.4f}")
        print("=" * 70)
        
        # Print classification report with explicit labels
        print("\nPer-Class Performance:")
        print(classification_report(
            all_y_true, all_y_pred,
            target_names=['Wake', 'N1', 'N2', 'N3', 'REM'],
            labels=all_labels,
            zero_division=0,
            digits=4
        ))
        
        # Save confusion matrix
        self.save_confusion_matrix(all_y_true, all_y_pred)
        
        return df_results, all_y_true, all_y_pred
    
    def save_confusion_matrix(self, y_true: List, y_pred: List, save_path: str = "./results/confusion_matrix.png"):
        """
        Generate and save confusion matrix plot.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            save_path: Path to save plot
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Define all possible labels
        all_labels = [0, 1, 2, 3, 4]
        
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Wake', 'N1', 'N2', 'N3', 'REM']
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title('Confusion Matrix - Sleep Stage Classification', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
        plt.close()


def main():
    """Main evaluation function."""
    
    # Configuration
    CHECKPOINT_PATH = "./weights/shhs/ours_diverse.pt"
    OUTPUT_DIR = "./output"
    DEVICE = "cpu"  # Force CPU
    
    print("=" * 70)
    print("mulEEG Sleep Stage Classification - Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Data directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nError: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please ensure the pretrained weights are placed in ./saved_weights/")
        return
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"\nError: Output directory not found at {OUTPUT_DIR}")
        print("Please run preprocessing first to generate patient data.")
        return
    
    # Initialize evaluator
    evaluator = PatientEvaluator(CHECKPOINT_PATH, OUTPUT_DIR, DEVICE)
    
    # Run evaluation
    df_results, all_y_true, all_y_pred = evaluator.evaluate_all_patients()
    
    # Save results
    os.makedirs("./results", exist_ok=True)
    
    # Save per-patient results
    results_csv = "./results/per_patient_results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\nPer-patient results saved to: {results_csv}")
    
    # Save summary statistics
    summary_file = "./results/evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("mulEEG Sleep Stage Classification - Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Data directory: {OUTPUT_DIR}\n")
        f.write(f"Device: {DEVICE}\n\n")
        f.write(f"Total Patients: {len(df_results)}\n")
        f.write(f"Total Epochs: {len(all_y_true)}\n\n")
        
        # Define all possible labels for summary file
        all_labels = [0, 1, 2, 3, 4]
        
        f.write(f"Overall Accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}\n")
        f.write(f"Overall Macro F1: {f1_score(all_y_true, all_y_pred, average='macro', labels=all_labels, zero_division=0):.4f}\n")
        f.write(f"Overall Weighted F1: {f1_score(all_y_true, all_y_pred, average='weighted', labels=all_labels, zero_division=0):.4f}\n")
        f.write(f"Overall Kappa: {cohen_kappa_score(all_y_true, all_y_pred, labels=all_labels):.4f}\n")
        f.write(f"Overall Balanced Acc: {balanced_accuracy_score(all_y_true, all_y_pred):.4f}\n\n")
        f.write("=" * 70 + "\n")
        f.write("Per-Class Performance:\n")
        f.write("=" * 70 + "\n\n")
        f.write(classification_report(
            all_y_true, all_y_pred,
            target_names=['Wake', 'N1', 'N2', 'N3', 'REM'],
            labels=all_labels,
            zero_division=0,
            digits=4
        ))
        f.write("\n" + "=" * 70 + "\n")
        f.write("Per-Patient Statistics:\n")
        f.write("=" * 70 + "\n\n")
        f.write(df_results.describe().to_string())
    
    print(f"Evaluation summary saved to: {summary_file}")
    
    print("\n" + "=" * 70)
    print("Evaluation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
 