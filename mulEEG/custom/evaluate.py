#!/usr/bin/env python
# coding: utf-8
"""
mulEEG Inference & Evaluation Pipeline
=======================================
Loads a pretrained ft_loss model, runs inference on every preprocessed .npz
subject file, computes sleep-stage metrics, and reports / saves results.

Run from the mulEEG root:
    python custom/evaluate.py
"""

import os
import sys
import json
import logging
import datetime
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ── Make mulEEG root importable when running from custom/ ─────────────────────
_SCRIPT_DIR  = os.path.abspath(os.path.dirname(__file__))
_MULEEG_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _MULEEG_ROOT not in sys.path:
    sys.path.insert(0, _MULEEG_ROOT)

from models.model import ft_loss
from config import Config

# ─────────────────────────────────────────────────────────────────────────────
# ABSOLUTE PATHS
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH          = '/home/nummm/Documents/CEPP/mulEEG/weights/shhs/ours_diverse.pt'
PREPROCESSING_OUTPUT_DIR = '/home/nummm/Documents/CEPP/mulEEG/custom/preprocessing_output'
RESULTS_DIR              = '/home/nummm/Documents/CEPP/mulEEG/custom/results'

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
STAGE_NAMES  = ['Wake', 'N1', 'N2', 'N3', 'REM']   # index 0-4
ALL_LABELS   = [0, 1, 2, 3, 4]
BATCH_SIZE   = 64    # epochs per forward pass
UNKNOWN_LABEL = -1   # set in preprocess.py for unmapped stages

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

_log_ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = os.path.join(RESULTS_DIR, f'prediction_{_log_ts}.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(_log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. HARDWARE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> torch.device:
    """Detect and log available hardware."""
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     gpu_name  = torch.cuda.get_device_name(0)
    #     gpu_cap   = torch.cuda.get_device_capability(0)
    #     gpu_mem   = torch.cuda.get_device_properties(0).total_memory / 1024**3
    #     log.info(f"GPU detected  : {gpu_name}")
    #     log.info(f"Compute cap   : sm_{gpu_cap[0]}{gpu_cap[1]}")
    #     log.info(f"VRAM          : {gpu_mem:.1f} GB")
    # else:
    #     device = torch.device('cpu')
    #     log.info("No GPU detected – using CPU")
    device = torch.device('cpu')
    return device


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Build ft_loss (encoder + linear head) and load pretrained weights.
    Raises FileNotFoundError if the checkpoint doesn't exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading model from: {checkpoint_path}")
    config = Config()

    try:
        model = ft_loss(checkpoint_path, config, device)
    except Exception as e:
        log.error(f"Failed to instantiate ft_loss: {e}")
        raise

    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[INFO] Model weights loaded successfully from {os.path.basename(checkpoint_path)}")
    log.info(f"       Total parameters : {n_params:,}")
    log.info(f"       Running on       : {device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA INGEST & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a preprocessed .npz file.

    Expected keys: x (n_epochs, 3000), y (n_epochs,), fs
    Returns (x, y) with shape validation.
    """
    data   = np.load(npz_path, allow_pickle=True)
    x      = data['x']   # (n_epochs, seq_len)
    y      = data['y']   # (n_epochs,)
    fs     = float(data['fs']) if 'fs' in data else 100.0

    # Shape validation
    if x.ndim != 2:
        raise ValueError(f"Expected x.ndim == 2, got {x.ndim}. Shape: {x.shape}")
    if x.shape[1] != 3000:
        log.warning(
            f"  Expected 3000 samples/epoch (100Hz × 30s), "
            f"got {x.shape[1]}. Sampling rate in file: {fs} Hz"
        )
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Epoch count mismatch: x has {x.shape[0]}, y has {y.shape[0]}"
        )

    log.debug(f"  Loaded: x={x.shape}, y={y.shape}, fs={fs} Hz")
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    subject_id: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched inference over all epochs of one subject.

    Returns:
        predictions : (n_epochs,) integer class indices
        confidences : (n_epochs,) softmax probability of the winning class
    """
    n_epochs     = len(x)
    predictions  = np.empty(n_epochs, dtype=np.int32)
    confidences  = np.empty(n_epochs, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_epochs, batch_size):
            end       = min(start + batch_size, n_epochs)
            batch_np  = x[start:end]                        # (B, 3000)
            # Model expects (B, 1, seq_len)
            tensor    = torch.from_numpy(batch_np).unsqueeze(1).float().to(device)

            logits    = model(tensor)                        # (B, 5)
            probs     = torch.softmax(logits, dim=1)         # (B, 5)
            preds     = logits.argmax(dim=1)                 # (B,)

            batch_preds = preds.cpu().numpy()
            batch_conf  = probs[torch.arange(len(preds)), preds].cpu().numpy()

            predictions[start:end] = batch_preds
            confidences[start:end] = batch_conf

            # Debug: log first epoch details
            if start == 0:
                ep0_logits = logits[0].cpu().numpy()
                ep0_probs  = probs[0].cpu().numpy()
                ep0_pred   = int(batch_preds[0])
                log.debug(
                    f"  [{subject_id}] Epoch-0 debug | "
                    f"Logits: {np.round(ep0_logits, 3)} | "
                    f"Probs: {np.round(ep0_probs, 3)} | "
                    f"Pred: {ep0_pred} ({STAGE_NAMES[ep0_pred]}, "
                    f"conf={batch_conf[0]:.4f})"
                )

    return predictions, confidences


# ─────────────────────────────────────────────────────────────────────────────
# 5. POST-PROCESSING & METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute standard sleep-staging metrics, ignoring UNKNOWN labels."""
    # Mask out UNKNOWN labels (set during preprocessing for unmapped stages)
    mask   = y_true != UNKNOWN_LABEL
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    per_f1 = f1_score(y_true, y_pred, average=None,
                      labels=ALL_LABELS, zero_division=0)

    return {
        'n_epochs':     int(len(y_true)),
        'accuracy':     float(accuracy_score(y_true, y_pred)),
        'macro_f1':     float(f1_score(y_true, y_pred, average='macro',
                                       labels=ALL_LABELS, zero_division=0)),
        'weighted_f1':  float(f1_score(y_true, y_pred, average='weighted',
                                       labels=ALL_LABELS, zero_division=0)),
        'kappa':        float(cohen_kappa_score(y_true, y_pred, labels=ALL_LABELS)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'f1_wake':      float(per_f1[0]),
        'f1_n1':        float(per_f1[1]),
        'f1_n2':        float(per_f1[2]),
        'f1_n3':        float(per_f1[3]),
        'f1_rem':       float(per_f1[4]),
    }


def log_epoch_predictions(
    subject_id: str,
    predictions: np.ndarray,
    confidences: np.ndarray,
    y_true: np.ndarray,
) -> None:
    """Log per-epoch predictions at DEBUG level."""
    for i, (pred, conf, truth) in enumerate(zip(predictions, confidences, y_true)):
        stage  = STAGE_NAMES[pred] if 0 <= pred <= 4 else f"UNKNOWN({pred})"
        truth_name = STAGE_NAMES[truth] if 0 <= truth <= 4 else f"UNKNOWN({truth})"
        match  = "✓" if pred == truth else "✗"
        log.debug(
            f"  [{subject_id}] Epoch {i+1:04d}: "
            f"Pred={stage:<3}  Conf={conf:.4f}  "
            f"True={truth_name:<3}  {match}"
        )


def stage_distribution(labels: np.ndarray, label_array: str = "pred") -> str:
    unique, counts = np.unique(labels[labels != UNKNOWN_LABEL], return_counts=True)
    parts = []
    for lbl, cnt in zip(unique, counts):
        name = STAGE_NAMES[lbl] if 0 <= lbl <= 4 else str(lbl)
        parts.append(f"{name}={cnt}")
    return f"{label_array}[" + ", ".join(parts) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: str,
    title: str = "Sleep Stage Classification – Confusion Matrix",
) -> None:
    cm   = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=STAGE_NAMES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(model: torch.nn.Module, device: torch.device) -> None:
    preproc_dir = Path(PREPROCESSING_OUTPUT_DIR)
    if not preproc_dir.exists():
        log.error(f"Preprocessing output directory not found: {preproc_dir}")
        log.error("Please run  custom/preprocess.py  first.")
        sys.exit(1)

    # Collect all .npz files (flat layout: preprocessing_output/<subject_id>.npz)
    npz_files = sorted(preproc_dir.glob("*.npz"))
    log.info(f"Found {len(npz_files)} .npz file(s) in {preproc_dir}")
    if not npz_files:
        log.warning("No .npz files found. Exiting.")
        return

    per_subject_rows: List[Dict] = []
    all_y_true:  List[int] = []
    all_y_pred:  List[int] = []
    all_y_conf:  List[float] = []

    log.info("=" * 70)
    for i, npz_path in enumerate(tqdm(npz_files, desc="Evaluating subjects", unit="subj"), 1):
        subject_id = npz_path.stem
        log.info(f"[{i:03d}/{len(npz_files):03d}] Subject: {subject_id}")

        # ── Load ──────────────────────────────────────────────────────────────
        try:
            x, y_true = load_npz(str(npz_path))
        except Exception as e:
            log.error(f"  [SKIP] Load failed: {e}")
            per_subject_rows.append({'subject_id': subject_id, 'status': 'LOAD_ERROR',
                                     'error': str(e)})
            continue

        # ── Infer ─────────────────────────────────────────────────────────────
        try:
            y_pred, y_conf = run_inference(model, x, device, subject_id=subject_id)
        except Exception as e:
            log.error(f"  [SKIP] Inference failed: {e}")
            per_subject_rows.append({'subject_id': subject_id, 'status': 'INFER_ERROR',
                                     'error': str(e)})
            continue

        # Per-epoch log
        log_epoch_predictions(subject_id, y_pred, y_conf, y_true)

        # Distribution info
        log.info(f"  {stage_distribution(y_true, 'true')}")
        log.info(f"  {stage_distribution(y_pred, 'pred')}")
        log.info(f"  Mean confidence : {y_conf.mean():.4f}")

        # ── Metrics ───────────────────────────────────────────────────────────
        try:
            metrics = compute_metrics(y_true, y_pred)
        except Exception as e:
            log.error(f"  [SKIP] Metrics failed: {e}")
            per_subject_rows.append({'subject_id': subject_id, 'status': 'METRIC_ERROR',
                                     'error': str(e)})
            continue

        metrics['subject_id'] = subject_id
        metrics['status']     = 'OK'
        per_subject_rows.append(metrics)

        log.info(
            f"  Accuracy={metrics['accuracy']:.4f}  "
            f"MacroF1={metrics['macro_f1']:.4f}  "
            f"Kappa={metrics['kappa']:.4f}  "
            f"BalAcc={metrics['balanced_acc']:.4f}"
        )

        # Save per-subject prediction CSV
        subj_csv_dir = os.path.join(RESULTS_DIR, 'per_subject')
        os.makedirs(subj_csv_dir, exist_ok=True)
        pd.DataFrame({
            'epoch':      np.arange(1, len(y_pred) + 1),
            'y_true':     y_true,
            'y_true_name':[STAGE_NAMES[t] if 0 <= t <= 4 else 'UNK' for t in y_true],
            'y_pred':     y_pred,
            'y_pred_name':[STAGE_NAMES[p] if 0 <= p <= 4 else 'UNK' for p in y_pred],
            'confidence': np.round(y_conf, 6),
        }).to_csv(os.path.join(subj_csv_dir, f'{subject_id}_predictions.csv'), index=False)

        # Accumulate for overall metrics
        mask_ok = y_true != UNKNOWN_LABEL
        all_y_true.extend(y_true[mask_ok].tolist())
        all_y_pred.extend(y_pred[mask_ok].tolist())
        all_y_conf.extend(y_conf[mask_ok].tolist())

    # ── Overall metrics ───────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("OVERALL EVALUATION RESULTS")
    log.info("=" * 70)

    ok_rows  = [r for r in per_subject_rows if r.get('status') == 'OK']
    err_rows = [r for r in per_subject_rows if r.get('status') != 'OK']

    log.info(f"Total .npz files    : {len(npz_files)}")
    log.info(f"Evaluated (OK)      : {len(ok_rows)}")
    log.info(f"Errors / skipped    : {len(err_rows)}")
    log.info(f"Total epochs used   : {len(all_y_true)}")

    if all_y_true:
        overall = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
        log.info(f"Overall Accuracy    : {overall['accuracy']:.4f}")
        log.info(f"Overall Macro F1    : {overall['macro_f1']:.4f}")
        log.info(f"Overall Weighted F1 : {overall['weighted_f1']:.4f}")
        log.info(f"Overall Kappa       : {overall['kappa']:.4f}")
        log.info(f"Overall Balanced Acc: {overall['balanced_acc']:.4f}")
        log.info(f"Mean confidence     : {np.mean(all_y_conf):.4f}")
        log.info("")
        log.info("Per-Class Performance:")
        log.info("\n" + classification_report(
            all_y_true, all_y_pred,
            target_names=STAGE_NAMES,
            labels=ALL_LABELS,
            zero_division=0,
            digits=4,
        ))
        log.info("=" * 70)

        # Confusion matrix PNG
        save_confusion_matrix(
            all_y_true, all_y_pred,
            save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png'),
        )

    # ── Save per-subject CSV ──────────────────────────────────────────────────
    df_summary = pd.DataFrame(per_subject_rows)
    csv_path   = os.path.join(RESULTS_DIR, 'per_subject_results.csv')
    df_summary.to_csv(csv_path, index=False)
    log.info(f"Per-subject CSV     → {csv_path}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    json_summary = {
        'run_timestamp':   _log_ts,
        'checkpoint':      CHECKPOINT_PATH,
        'preprocessing_dir': PREPROCESSING_OUTPUT_DIR,
        'n_subjects_total': len(npz_files),
        'n_subjects_ok':    len(ok_rows),
        'n_epochs_total':   len(all_y_true),
        'overall_metrics':  overall if all_y_true else {},
        'per_subject': [
            {k: v for k, v in r.items() if k != 'status'}
            for r in ok_rows
        ],
    }
    json_path = os.path.join(RESULTS_DIR, 'evaluation_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)
    log.info(f"JSON summary        → {json_path}")
    log.info(f"Log file            → {_log_file}")
    log.info("")
    log.info("Evaluation completed successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    log.info("=" * 70)
    log.info("mulEEG Sleep Stage Classification – Inference Pipeline")
    log.info("=" * 70)
    log.info(f"Checkpoint          : {CHECKPOINT_PATH}")
    log.info(f"Preprocessing dir   : {PREPROCESSING_OUTPUT_DIR}")
    log.info(f"Results dir         : {RESULTS_DIR}")
    log.info(f"Batch size          : {BATCH_SIZE} epochs/forward-pass")
    log.info("=" * 70)

    # 1. Hardware
    device = detect_device()

    # 2. Model
    model = load_model(CHECKPOINT_PATH, device)

    # 3-7. Evaluate all subjects
    evaluate_all(model, device)
