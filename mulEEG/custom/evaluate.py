"""
Evaluation script for mulEEG sleep stage classification model.

Folder structure expected:
    mulEEG/custom/preprocessing_output/
    ├── [patient_id_1]/
    │   └── [patient_id_1].npz
    └── [patient_id_2]/
        └── [patient_id_2].npz

Results are saved to:
    mulEEG/custom/results/
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# ── Ensure mulEEG root is on sys.path so imports work when running from custom/ ──
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_MULEEG_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _MULEEG_ROOT not in sys.path:
    sys.path.insert(0, _MULEEG_ROOT)

# Import model and config (from mulEEG root)
from models.model import ft_loss
from config import Config


# ==================== PATHS (absolute) ====================
# Pretrained checkpoint
CHECKPOINT_PATH = os.path.abspath(
    os.path.join(_MULEEG_ROOT, 'weights', 'shhs', 'ours_diverse.pt')
)

# Preprocessed patient data (output of preprocess.py)
PREPROCESSING_OUTPUT_DIR = os.path.abspath(
    os.path.join(_SCRIPT_DIR, 'preprocessing_output')
)

# Where to save evaluation results
RESULTS_DIR = os.path.abspath(
    os.path.join(_SCRIPT_DIR, 'results')
)
# ==========================================================


class PatientEvaluator:
    """Evaluates pretrained mulEEG model on preprocessed patient data."""

    def __init__(self, checkpoint_path: str, preprocessing_output_dir: str,
                 device: str = "cpu"):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Absolute path to pretrained model checkpoint (.pt)
            preprocessing_output_dir: Absolute path to preprocessing_output folder
            device: Torch device string (default: "cpu")
        """
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(preprocessing_output_dir)
        self.device = torch.device(device)

        # Config
        self.config = Config()
        self.config.device = device

        # Load model
        print(f"Loading model from: {checkpoint_path}")
        self.model = ft_loss(checkpoint_path, self.config, device)
        self.model.eval()
        self.model.to(self.device)
        print(f"Model loaded successfully on {device}")

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_patient_data(self, patient_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load patient data from .npz file.

        Args:
            patient_path: Path to patient folder inside preprocessing_output

        Returns:
            Tuple of (x, y) where x has shape (n_epochs, seq_len) and y (n_epochs,)

        Raises:
            FileNotFoundError: If the .npz file does not exist
        """
        patient_id = patient_path.name
        npz_file = patient_path / f"{patient_id}.npz"

        if not npz_file.exists():
            raise FileNotFoundError(f".npz not found: {npz_file}")

        data = np.load(npz_file)
        x = data['x']   # (n_epochs, sequence_length)
        y = data['y']   # (n_epochs,)
        return x, y

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_patient(self, x: np.ndarray, debug: bool = True) -> np.ndarray:
        """
        Run inference on a single patient's data, one epoch at a time
        (no batching – low-RAM mode).

        Args:
            x: Input data, shape (n_epochs, sequence_length)
            debug: Print debug info for the first epoch

        Returns:
            Predictions array, shape (n_epochs,)
        """
        predictions = []

        with torch.no_grad():
            for i in range(len(x)):
                # Shape: (1, 1, sequence_length)
                epoch_tensor = torch.from_numpy(x[i:i+1]).unsqueeze(1).float().to(self.device)
                output = self.model(epoch_tensor)   # (1, n_classes)

                if debug and i == 0:
                    logit = output[0].cpu().numpy()
                    probs = torch.softmax(output[0], dim=0).cpu().numpy()
                    pred = int(logit.argmax())
                    print(f"\n  🔍 Debug (epoch 0):")
                    print(f"     Logits : {logit}")
                    print(f"     Probs  : {probs}")
                    print(f"     Pred   : {pred} (conf {probs[pred]:.4f})")

                pred = int(output.argmax(dim=1).cpu().numpy()[0])
                predictions.append(pred)

        return np.array(predictions)

    # ── Per-patient evaluation ────────────────────────────────────────────────

    def evaluate_patient(self, patient_path: Path) -> Tuple[Optional[Dict], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Evaluate model on a single patient.

        Args:
            patient_path: Path to patient folder

        Returns:
            (metrics dict, y_true, y_pred) or (None, None, None) on error
        """
        patient_id = patient_path.name
        all_labels = [0, 1, 2, 3, 4]

        try:
            x, y_true = self.load_patient_data(patient_path)
            y_pred = self.predict_patient(x)

            print(f"  Labels in y_true: {np.unique(y_true)}")
            print(f"  Labels in y_pred: {np.unique(y_pred)}")

            per_class_f1 = f1_score(y_true, y_pred, average=None,
                                    labels=all_labels, zero_division=0)

            metrics = {
                'patient_id':   patient_id,
                'n_epochs':     len(y_true),
                'accuracy':     accuracy_score(y_true, y_pred),
                'macro_f1':     f1_score(y_true, y_pred, average='macro',
                                         labels=all_labels, zero_division=0),
                'weighted_f1':  f1_score(y_true, y_pred, average='weighted',
                                         labels=all_labels, zero_division=0),
                'kappa':        cohen_kappa_score(y_true, y_pred, labels=all_labels),
                'balanced_acc': balanced_accuracy_score(y_true, y_pred),
                'f1_wake':      per_class_f1[0],
                'f1_n1':        per_class_f1[1],
                'f1_n2':        per_class_f1[2],
                'f1_n3':        per_class_f1[3],
                'f1_rem':       per_class_f1[4],
            }
            return metrics, y_true, y_pred

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    # ── Batch evaluation over all patients ────────────────────────────────────

    def evaluate_all_patients(self) -> Tuple[pd.DataFrame, List, List]:
        """
        Evaluate model on every patient folder found in preprocessing_output.

        Skips folders that have no .npz file (e.g. patients whose EDF was missing
        during preprocessing).

        Returns:
            (df_results, all_y_true, all_y_pred)
        """
        if not self.output_dir.exists():
            print(f"ERROR: Preprocessing output directory not found: {self.output_dir}")
            return pd.DataFrame(), [], []

        patient_folders = sorted(p for p in self.output_dir.iterdir() if p.is_dir())
        print(f"\nFound {len(patient_folders)} patient folder(s) in preprocessing_output")
        print("=" * 70)

        results = []
        all_y_true: List[int] = []
        all_y_pred: List[int] = []
        skipped = 0

        for i, patient_path in enumerate(patient_folders, 1):
            patient_id = patient_path.name
            npz_file = patient_path / f"{patient_id}.npz"

            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(patient_folders)}] Patient: {patient_id}")

            # Skip if .npz missing (preprocessing was skipped for this patient)
            if not npz_file.exists():
                print(f"  ⊘ .npz not found – skipping")
                skipped += 1
                continue

            metrics, y_true, y_pred = self.evaluate_patient(patient_path)

            if metrics is not None:
                results.append(metrics)
                all_y_true.extend(y_true.tolist())
                all_y_pred.extend(y_pred.tolist())

                print(f"  Epochs     : {metrics['n_epochs']}")
                print(f"  Accuracy   : {metrics['accuracy']:.4f}")
                print(f"  Macro F1   : {metrics['macro_f1']:.4f}")
                print(f"  Kappa      : {metrics['kappa']:.4f}")

        df_results = pd.DataFrame(results)
        all_labels = [0, 1, 2, 3, 4]

        # ── Overall metrics ───────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("OVERALL EVALUATION RESULTS")
        print("=" * 70)
        print(f"Total patient folders : {len(patient_folders)}")
        print(f"Evaluated             : {len(results)}")
        print(f"Skipped (no .npz)     : {skipped}")
        print(f"Total epochs          : {len(all_y_true)}")

        if all_y_true:
            print(f"Overall Accuracy      : {accuracy_score(all_y_true, all_y_pred):.4f}")
            print(f"Overall Macro F1      : {f1_score(all_y_true, all_y_pred, average='macro', labels=all_labels, zero_division=0):.4f}")
            print(f"Overall Weighted F1   : {f1_score(all_y_true, all_y_pred, average='weighted', labels=all_labels, zero_division=0):.4f}")
            print(f"Overall Kappa         : {cohen_kappa_score(all_y_true, all_y_pred, labels=all_labels):.4f}")
            print(f"Overall Balanced Acc  : {balanced_accuracy_score(all_y_true, all_y_pred):.4f}")
            print("=" * 70)

            print("\nPer-Class Performance:")
            print(classification_report(
                all_y_true, all_y_pred,
                target_names=['Wake', 'N1', 'N2', 'N3', 'REM'],
                labels=all_labels,
                zero_division=0,
                digits=4,
            ))

            # Save confusion matrix
            self.save_confusion_matrix(all_y_true, all_y_pred)

        return df_results, all_y_true, all_y_pred

    # ── Confusion matrix ──────────────────────────────────────────────────────

    def save_confusion_matrix(self, y_true: List, y_pred: List,
                               save_path: Optional[str] = None):
        """
        Generate and save confusion matrix plot.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            save_path: Absolute path to save the PNG (default: RESULTS_DIR/confusion_matrix.png)
        """
        if save_path is None:
            save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        all_labels = [0, 1, 2, 3, 4]
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Wake', 'N1', 'N2', 'N3', 'REM'],
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title('Confusion Matrix – Sleep Stage Classification', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to: {save_path}")
        plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Main evaluation function."""

    # ── Config ────────────────────────────────────────────────────────────────
    checkpoint_path        = CHECKPOINT_PATH
    preprocessing_out_dir  = PREPROCESSING_OUTPUT_DIR
    results_dir            = RESULTS_DIR
    device                 = "cpu"   # Force CPU (low-RAM mode)
    # ─────────────────────────────────────────────────────────────────────────

    print("=" * 70)
    print("mulEEG Sleep Stage Classification – Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Preprocessing dir: {preprocessing_out_dir}")
    print(f"Results dir      : {results_dir}")
    print(f"Device           : {device}")
    print("=" * 70)

    # Validate checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        return

    # Validate preprocessing output directory
    if not os.path.exists(preprocessing_out_dir):
        print(f"\nERROR: Preprocessing output directory not found: {preprocessing_out_dir}")
        print("Please run custom/preprocess.py first.")
        return

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluator = PatientEvaluator(checkpoint_path, preprocessing_out_dir, device)
    df_results, all_y_true, all_y_pred = evaluator.evaluate_all_patients()

    if df_results.empty:
        print("\nNo results to save.")
        return

    all_labels = [0, 1, 2, 3, 4]

    # ── Save per-patient CSV ──────────────────────────────────────────────────
    results_csv = os.path.join(results_dir, 'per_patient_results.csv')
    df_results.to_csv(results_csv, index=False)
    print(f"\nPer-patient results saved to: {results_csv}")

    # ── Save summary text ─────────────────────────────────────────────────────
    summary_file = os.path.join(results_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("mulEEG Sleep Stage Classification – Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint       : {checkpoint_path}\n")
        f.write(f"Preprocessing dir: {preprocessing_out_dir}\n")
        f.write(f"Device           : {device}\n\n")
        f.write(f"Total Patients   : {len(df_results)}\n")
        f.write(f"Total Epochs     : {len(all_y_true)}\n\n")
        f.write(f"Overall Accuracy    : {accuracy_score(all_y_true, all_y_pred):.4f}\n")
        f.write(f"Overall Macro F1    : {f1_score(all_y_true, all_y_pred, average='macro', labels=all_labels, zero_division=0):.4f}\n")
        f.write(f"Overall Weighted F1 : {f1_score(all_y_true, all_y_pred, average='weighted', labels=all_labels, zero_division=0):.4f}\n")
        f.write(f"Overall Kappa       : {cohen_kappa_score(all_y_true, all_y_pred, labels=all_labels):.4f}\n")
        f.write(f"Overall Balanced Acc: {balanced_accuracy_score(all_y_true, all_y_pred):.4f}\n\n")
        f.write("=" * 70 + "\n")
        f.write("Per-Class Performance:\n")
        f.write("=" * 70 + "\n\n")
        f.write(classification_report(
            all_y_true, all_y_pred,
            target_names=['Wake', 'N1', 'N2', 'N3', 'REM'],
            labels=all_labels,
            zero_division=0,
            digits=4,
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