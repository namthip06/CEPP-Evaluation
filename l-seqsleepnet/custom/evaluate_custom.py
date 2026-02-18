#!/usr/bin/env python3
"""
L-SeqSleepNet — Evaluation Script
===================================
Reads per-subject test_ret.mat files from custom/results/ and evaluates
sleep staging performance using the same aggregation logic as evaluate.py.

Aggregation (from evaluate.py):
  score shape in test_ret.mat: (N_valid, SEQ_LEN, 5)
  → transpose to (SEQ_LEN, N_valid, 5)
  → roll overlapping windows → argmax → final prediction per epoch

Metrics per subject and overall:
  - Accuracy
  - F1-score per class (W, N1, N2, N3, REM) + macro mean
  - Cohen's Kappa
  - Sensitivity (macro)
  - Specificity (macro)

Output:
  custom/evaluation/
      per_subject_metrics.csv   ← one row per subject
      overall_metrics.json      ← aggregated across all subjects
      confusion_matrix.csv      ← overall confusion matrix

Constraints:
  - No batch / no parallel processing (low RAM)
  - Absolute paths throughout
  - Skip subjects without result files
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import h5py
import hdf5storage

from sklearn import metrics as sk_metrics

# ── Absolute paths ────────────────────────────────────────────────────────────
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PREPROC_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), 'preprocessing_output'))
RESULTS_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
EVAL_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), 'evaluation'))

# ── Model config ──────────────────────────────────────────────────────────────
SUB_SEQ_LEN  = 10
NSUBSEQ      = 20
SEQ_LEN      = SUB_SEQ_LEN * NSUBSEQ   # 200
NCLASS       = 5
AGGREGATION  = 'multiplication'   # 'multiplication' or 'average'

# Sleep stage names (1-indexed labels from model output)
STAGE_NAMES  = {1: 'W', 2: 'N1', 3: 'N2', 4: 'N3', 5: 'REM'}
STAGE_LABELS = [1, 2, 3, 4, 5]


# ─────────────────────────────────────────────────────────────────────────────
def softmax(z):
    """Numerically stable softmax, input shape (N, C)."""
    s = np.max(z, axis=1, keepdims=True)
    e_x = np.exp(z - s)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def aggregate_avg(score):
    """
    Average aggregation over overlapping windows.

    Parameters
    ----------
    score : np.ndarray, shape (SEQ_LEN, N_valid, NCLASS)
        Transposed score from test_ret.mat.

    Returns
    -------
    pred : np.ndarray, shape (N_valid + SEQ_LEN - 1,)
        1-indexed predicted labels for all epochs.
    """
    N_valid = score.shape[1]
    total   = N_valid + SEQ_LEN - 1
    fused   = None

    for i in range(SEQ_LEN):
        prob_i = np.concatenate(
            [np.zeros((SEQ_LEN - 1, NCLASS)), softmax(score[i])],
            axis=0
        )
        prob_i = np.roll(prob_i, -(SEQ_LEN - i - 1), axis=0)
        fused  = prob_i if fused is None else fused + prob_i

    return np.argmax(fused, axis=-1) + 1   # 1-indexed


def aggregate_mul(score):
    """
    Multiplication (log-sum) aggregation over overlapping windows.

    Parameters
    ----------
    score : np.ndarray, shape (SEQ_LEN, N_valid, NCLASS)

    Returns
    -------
    pred : np.ndarray, shape (N_valid + SEQ_LEN - 1,)
        1-indexed predicted labels for all epochs.
    """
    N_valid = score.shape[1]
    total   = N_valid + SEQ_LEN - 1
    fused   = None

    for i in range(SEQ_LEN):
        prob_i = np.log10(softmax(score[i]) + 1e-10)
        prob_i = np.concatenate(
            [np.ones((SEQ_LEN - 1, NCLASS)), prob_i],
            axis=0
        )
        prob_i = np.roll(prob_i, -(SEQ_LEN - i - 1), axis=0)
        fused  = prob_i if fused is None else fused + prob_i

    return np.argmax(fused, axis=-1) + 1   # 1-indexed


def aggregate(score, method='multiplication'):
    """Dispatch to avg or mul aggregation."""
    if method == 'average':
        return aggregate_avg(score)
    return aggregate_mul(score)


# ─────────────────────────────────────────────────────────────────────────────
def load_groundtruth(mat_path):
    """
    Load ground-truth labels from preprocessing .mat file.

    Returns
    -------
    label : np.ndarray, shape (N,)   — 0-indexed
    """
    with h5py.File(mat_path, 'r') as f:
        lbl = np.array(f['label'], dtype=np.float32)   # (1, N)
    return np.squeeze(lbl).astype(int)   # (N,)


def load_result(result_path):
    """
    Load score from test_ret.mat.

    Returns
    -------
    score : np.ndarray, shape (SEQ_LEN, N_valid, NCLASS)
    label : np.ndarray, shape (N,)   — 0-indexed ground truth stored in result
    """
    data  = hdf5storage.loadmat(file_name=result_path)
    score = np.array(data['score'])    # (N_valid, SEQ_LEN, NCLASS)
    label = np.array(data['label'])    # (N,) — 0-indexed

    # Transpose score: (N_valid, SEQ_LEN, NCLASS) → (SEQ_LEN, N_valid, NCLASS)
    score = np.transpose(score, (1, 0, 2))
    return score, label.astype(int)


# ─────────────────────────────────────────────────────────────────────────────
def calculate_metrics(y_true_1idx, y_pred_1idx):
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_true_1idx : np.ndarray  — 1-indexed ground truth
    y_pred_1idx : np.ndarray  — 1-indexed predictions

    Returns
    -------
    ret : dict
    """
    try:
        from imblearn.metrics import sensitivity_score, specificity_score
        sens = float(sensitivity_score(y_true_1idx, y_pred_1idx,
                                       labels=STAGE_LABELS, average='macro'))
        spec = float(specificity_score(y_true_1idx, y_pred_1idx,
                                       labels=STAGE_LABELS, average='macro'))
    except ImportError:
        # imblearn not installed — compute manually
        sens, spec = compute_sens_spec_manual(y_true_1idx, y_pred_1idx)

    f1_per_class = sk_metrics.f1_score(
        y_true_1idx, y_pred_1idx,
        labels=STAGE_LABELS, average=None, zero_division=0
    )

    ret = {
        'accuracy':    float(sk_metrics.accuracy_score(y_true_1idx, y_pred_1idx)),
        'kappa':       float(sk_metrics.cohen_kappa_score(
                           y_true_1idx, y_pred_1idx, labels=STAGE_LABELS)),
        'mean_f1':     float(np.mean(f1_per_class)),
        'f1_W':        float(f1_per_class[0]),
        'f1_N1':       float(f1_per_class[1]),
        'f1_N2':       float(f1_per_class[2]),
        'f1_N3':       float(f1_per_class[3]),
        'f1_REM':      float(f1_per_class[4]),
        'sensitivity': sens,
        'specificity': spec,
        'n_epochs':    int(len(y_true_1idx)),
    }
    return ret


def compute_sens_spec_manual(y_true, y_pred):
    """Fallback: compute macro sensitivity/specificity without imblearn."""
    sens_list, spec_list = [], []
    for cls in STAGE_LABELS:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        tn = np.sum((y_true != cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(sens_list)), float(np.mean(spec_list))


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subject(subject_id):
    """
    Evaluate one subject.

    Returns
    -------
    metrics_dict : dict or None
    y_true_1idx  : np.ndarray or None
    y_pred_1idx  : np.ndarray or None
    """
    result_path = os.path.join(RESULTS_DIR, f'{subject_id}_test_ret.mat')
    mat_path    = os.path.join(PREPROC_DIR, f'{subject_id}_eeg.mat')

    if not os.path.exists(result_path):
        print(f"  Skip {subject_id}: no test_ret.mat")
        return None, None, None

    if not os.path.exists(mat_path):
        print(f"  Skip {subject_id}: no _eeg.mat (ground truth)")
        return None, None, None

    try:
        # Load score and ground truth
        score, lbl_result = load_result(result_path)   # score: (SEQ_LEN, N_valid, 5)
        lbl_preproc       = load_groundtruth(mat_path)  # (N,) 0-indexed

        N       = len(lbl_preproc)
        N_valid = score.shape[1]

        # Aggregate scores → predictions (1-indexed, length = N_valid + SEQ_LEN - 1)
        pred_all = aggregate(score, method=AGGREGATION)   # shape: (N,)

        # Ground truth: convert 0-indexed → 1-indexed to match predictions
        y_true = lbl_preproc + 1   # (N,) 1-indexed

        # pred_all covers all N epochs (N_valid + SEQ_LEN - 1 = N)
        y_pred = pred_all[:N]

        # Sanity check
        if len(y_true) != len(y_pred):
            min_len = min(len(y_true), len(y_pred))
            y_true  = y_true[:min_len]
            y_pred  = y_pred[:min_len]

        m = calculate_metrics(y_true, y_pred)
        m['subject_id'] = subject_id

        print(f"  acc={m['accuracy']:.3f}  kappa={m['kappa']:.3f}  "
              f"mean_F1={m['mean_f1']:.3f}  n={m['n_epochs']}")

        del score
        return m, y_true, y_pred

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("L-SeqSleepNet — Evaluation")
    print("=" * 70)
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Preproc dir : {PREPROC_DIR}")
    print(f"Output dir  : {EVAL_DIR}")
    print(f"Aggregation : {AGGREGATION}")
    print("=" * 70)

    os.makedirs(EVAL_DIR, exist_ok=True)

    # ── Collect subject IDs from results directory ────────────────────────────
    result_files = sorted([
        f for f in os.listdir(RESULTS_DIR)
        if f.endswith('_test_ret.mat')
    ])
    subject_ids = [f.replace('_test_ret.mat', '') for f in result_files]
    print(f"\nFound {len(subject_ids)} result files\n")

    if not subject_ids:
        print("No results to evaluate.")
        sys.exit(0)

    # ── Evaluate each subject ─────────────────────────────────────────────────
    all_metrics   = []
    all_y_true    = []
    all_y_pred    = []

    for i, sid in enumerate(subject_ids, 1):
        print(f"[{i}/{len(subject_ids)}] {sid}")
        m, y_true, y_pred = evaluate_subject(sid)

        if m is not None:
            all_metrics.append(m)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)

    if not all_metrics:
        print("\nNo subjects evaluated successfully.")
        sys.exit(0)

    # ── Per-subject CSV ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_metrics)
    cols = ['subject_id', 'n_epochs', 'accuracy', 'kappa', 'mean_f1',
            'f1_W', 'f1_N1', 'f1_N2', 'f1_N3', 'f1_REM',
            'sensitivity', 'specificity']
    df = df[[c for c in cols if c in df.columns]]
    per_subject_path = os.path.join(EVAL_DIR, 'per_subject_metrics.csv')
    df.to_csv(per_subject_path, index=False, float_format='%.4f')
    print(f"\nPer-subject CSV → {per_subject_path}")

    # ── Overall (aggregate all subjects) ─────────────────────────────────────
    y_true_all = np.hstack(all_y_true)
    y_pred_all = np.hstack(all_y_pred)
    overall    = calculate_metrics(y_true_all, y_pred_all)
    overall['n_subjects'] = len(all_metrics)

    overall_path = os.path.join(EVAL_DIR, 'overall_metrics.json')
    with open(overall_path, 'w') as f:
        json.dump(overall, f, indent=2)
    print(f"Overall JSON  → {overall_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = sk_metrics.confusion_matrix(y_true_all, y_pred_all, labels=STAGE_LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{STAGE_NAMES[l]}' for l in STAGE_LABELS],
        columns=[f'Pred_{STAGE_NAMES[l]}' for l in STAGE_LABELS],
    )
    cm_path = os.path.join(EVAL_DIR, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    print(f"Confusion CSV → {cm_path}")

    # ── Summary print ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Subjects evaluated : {overall['n_subjects']}")
    print(f"Total epochs       : {overall['n_epochs']}")
    print(f"Accuracy           : {overall['accuracy']:.4f}")
    print(f"Cohen's Kappa      : {overall['kappa']:.4f}")
    print(f"Mean F1            : {overall['mean_f1']:.4f}")
    print(f"  F1-W             : {overall['f1_W']:.4f}")
    print(f"  F1-N1            : {overall['f1_N1']:.4f}")
    print(f"  F1-N2            : {overall['f1_N2']:.4f}")
    print(f"  F1-N3            : {overall['f1_N3']:.4f}")
    print(f"  F1-REM           : {overall['f1_REM']:.4f}")
    print(f"Sensitivity (macro): {overall['sensitivity']:.4f}")
    print(f"Specificity (macro): {overall['specificity']:.4f}")
    print("=" * 70)

    # ── Per-subject summary ───────────────────────────────────────────────────
    print("\nPer-subject summary:")
    print(f"  Mean accuracy : {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
    print(f"  Mean kappa    : {df['kappa'].mean():.4f} ± {df['kappa'].std():.4f}")
    print(f"  Mean F1       : {df['mean_f1'].mean():.4f} ± {df['mean_f1'].std():.4f}")
    print(f"\nAll outputs saved to: {EVAL_DIR}")


if __name__ == '__main__':
    main()
