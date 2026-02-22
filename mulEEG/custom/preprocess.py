#!/usr/bin/env python
# coding: utf-8
"""
SHHS EEG Preprocessing Pipeline
================================
Reads raw EEG recordings from the rawEEG directory structure, processes each
subject's EDF + hypnogram CSV, and saves cleaned .npz files ready for model
training.

Directory layout expected under BASE_DIR:
  <subject_id>/
    ├── edf_signals.edf
    ├── csv_hypnogram.csv
    └── csv_events.csv   (optional – not used here)

Output layout under OUTPUT_BASE_DIR:
  <subject_id>.npz   →  keys: x (epochs), y (labels), fs (sampling rate),
                                source_edf, source_csv
"""

import os
from typing import List, Optional
import sys
import logging
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from mne.io import read_raw_edf
import mne

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (absolute)
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR     = os.path.abspath(os.path.dirname(__file__))
BASE_DIR        = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', 'rawEEG'))
OUTPUT_BASE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, 'preprocessing_output'))

# ─────────────────────────────────────────────────────────────────────────────
# HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SFREQ   = 100          # Hz  – downsample target
EPOCH_SEC      = 30           # seconds per epoch
SAMPLES_EPOCH  = TARGET_SFREQ * EPOCH_SEC   # 3 000 samples per epoch
MIN_DURATION_MIN = 60         # skip recordings shorter than this (minutes)
EEG_CHANNEL    = 'EEG C4-A1' # primary channel name to look for

# Sleep-stage mapping: CSV label string → integer class
STAGE_MAP = {
    'WK':  0, 'W':   0,
    'N1':  1, 'S1':  1,
    'N2':  2, 'S2':  2,
    'N3':  3, 'S3':  3, 'S4': 3, 'N4': 3,
    'REM': 4, 'R':   4,
}
UNKNOWN_LABEL = -1   # used when a stage string is not in STAGE_MAP

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

_log_ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = os.path.join(OUTPUT_BASE_DIR, f'preprocess_{_log_ts}.log')

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

# Silence MNE's verbose output so our own logs stay readable
mne.set_log_level('WARNING')

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def find_eeg_channel(ch_names: List[str], preferred: str = EEG_CHANNEL) -> Optional[str]:
    """
    Return the channel name that best matches *preferred*.
    Falls back to any channel starting with 'EEG C4' if an exact match fails.
    Returns None if nothing is found.
    """
    # Exact match
    if preferred in ch_names:
        log.debug(f"  Found exact channel '{preferred}'")
        return preferred

    # Case-insensitive exact match
    for ch in ch_names:
        if ch.strip().upper() == preferred.upper():
            log.debug(f"  Found case-insensitive channel '{ch}'")
            return ch

    # Partial match on the first part (e.g. 'EEG C4')
    prefix = preferred.split('-')[0].strip().upper()
    candidates = [ch for ch in ch_names if ch.strip().upper().startswith(prefix)]
    if candidates:
        log.debug(f"  Partial match – using '{candidates[0]}' (candidates: {candidates})")
        return candidates[0]

    log.warning(f"  Channel '{preferred}' NOT found. Available: {ch_names}")
    return None


def read_hypnogram(csv_path: str) -> Optional[np.ndarray]:
    """
    Read the hypnogram CSV and return an integer label array.
    Expects a column named 'stage' (case-insensitive) or uses the first column.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log.error(f"  Cannot read hypnogram CSV: {e}")
        return None

    # Find the stage column
    stage_col = None
    for col in df.columns:
        if col.strip().lower() == 'sleep stage':
            stage_col = col
            break
    if stage_col is None:
        stage_col = df.columns[0]
        log.debug(f"  No 'stage' column found; using first column '{stage_col}'")

    raw_stages = df[stage_col].astype(str).str.strip().tolist()
    labels = []
    unknown_set = set()

    for s in raw_stages:
        mapped = STAGE_MAP.get(s, STAGE_MAP.get(s.upper(), None))
        if mapped is None:
            unknown_set.add(s)
            labels.append(UNKNOWN_LABEL)
        else:
            labels.append(mapped)

    if unknown_set:
        log.critical(
            f"  Unknown stage label(s) encountered: {unknown_set}. "
            f"Mapped to UNKNOWN_LABEL ({UNKNOWN_LABEL})."
        )

    return np.array(labels, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def process_subject(subject_dir: str) -> dict:
    """
    Process one subject folder.  Returns a result dict with fields:
        subject, status, n_epochs, reason
    """
    subject_id = os.path.basename(subject_dir)
    result = dict(subject=subject_id, status='ERROR', n_epochs=0, reason='')

    edf_path = os.path.join(subject_dir, 'edf_signals.edf')
    csv_path = os.path.join(subject_dir, 'csv_hypnogram.csv')

    # ── 1. File presence check ────────────────────────────────────────────────
    missing = [f for f in [edf_path, csv_path] if not os.path.isfile(f)]
    if missing:
        msg = f"Missing file(s): {[os.path.basename(m) for m in missing]}. Skipping."
        log.error(f"[{subject_id}] {msg}")
        result['reason'] = msg
        return result

    log.info(f"[{subject_id}] ─── Starting ───────────────────────────────────")

    # ── 2. EDF header integrity & sampling rate ───────────────────────────────
    try:
        raw = read_raw_edf(edf_path, preload=False, stim_channel=None, verbose=False)
    except Exception as e:
        msg = f"Failed to open EDF: {e}"
        log.error(f"[{subject_id}] {msg}")
        result['reason'] = msg
        return result

    orig_sfreq = raw.info['sfreq']
    ch_names   = raw.info['ch_names']
    n_times    = raw.n_times
    duration_s = n_times / orig_sfreq
    duration_min = duration_s / 60.0

    log.info(
        f"[{subject_id}] Original SFreq: {orig_sfreq:.1f} Hz | "
        f"Channels: {len(ch_names)} | Duration: {duration_min:.1f} min"
    )

    # ── 3. Duration filter ─────────────────────────────────────────────────────
    if duration_min < MIN_DURATION_MIN:
        msg = (
            f"Duration {duration_min:.1f} min < {MIN_DURATION_MIN} min threshold. "
            "Recording too short for sleep-architecture learning. Skipping."
        )
        log.warning(f"[{subject_id}] {msg}")
        result['reason'] = msg
        return result

    # ── 4. Channel selection ──────────────────────────────────────────────────
    ch = find_eeg_channel(ch_names, preferred=EEG_CHANNEL)
    if ch is None:
        msg = f"Required channel '{EEG_CHANNEL}' not found. Skipping."
        log.error(f"[{subject_id}] {msg}")
        result['reason'] = msg
        return result

    ch_idx = ch_names.index(ch)
    log.debug(f"[{subject_id}] Using channel '{ch}' at index {ch_idx}")

    # ── 5. Load & resample ────────────────────────────────────────────────────
    try:
        raw.load_data()
        raw.pick([ch])
        if orig_sfreq != TARGET_SFREQ:
            log.info(
                f"[{subject_id}] Resampling {orig_sfreq:.1f} Hz → {TARGET_SFREQ} Hz …"
            )
            raw.resample(TARGET_SFREQ, npad='auto')
        signal = raw.get_data()[0]   # shape: (n_samples,)
    except Exception as e:
        msg = f"Signal loading/resampling failed: {e}"
        log.error(f"[{subject_id}] {msg}")
        result['reason'] = msg
        return result

    log.debug(f"[{subject_id}] Signal shape after resample: {signal.shape}")

    # ── 6. Epoch divisibility check & trim ────────────────────────────────────
    remainder = len(signal) % SAMPLES_EPOCH
    if remainder != 0:
        trimmed = len(signal) - remainder
        log.warning(
            f"[{subject_id}] Signal length {len(signal)} not divisible by "
            f"{SAMPLES_EPOCH} ({EPOCH_SEC}s × {TARGET_SFREQ}Hz). "
            f"Trimming {remainder} trailing samples → {trimmed} samples."
        )
        signal = signal[:trimmed]

    n_epochs_from_edf = len(signal) // SAMPLES_EPOCH
    log.info(f"[{subject_id}] EDF epochs: {n_epochs_from_edf}")

    # ── 7. Read hypnogram labels ──────────────────────────────────────────────
    labels = read_hypnogram(csv_path)
    if labels is None:
        msg = "Hypnogram read failed. Skipping."
        log.error(f"[{subject_id}] {msg}")
        result['reason'] = msg
        return result

    n_epochs_from_csv = len(labels)
    log.info(f"[{subject_id}] CSV rows (epochs): {n_epochs_from_csv}")

    # ── 8. Label / epoch count synchronisation ────────────────────────────────
    if n_epochs_from_edf != n_epochs_from_csv:
        log.warning(
            f"[{subject_id}] Mismatch! EDF: {n_epochs_from_edf} epochs, "
            f"CSV: {n_epochs_from_csv} rows. Adjusting to min …"
        )
        n_use = min(n_epochs_from_edf, n_epochs_from_csv)
        signal = signal[: n_use * SAMPLES_EPOCH]
        labels = labels[:n_use]
        log.info(f"[{subject_id}] Aligned to {n_use} epochs.")
    else:
        n_use = n_epochs_from_edf

    # ── 9. Reject UNKNOWN labels ──────────────────────────────────────────────
    if np.any(labels == UNKNOWN_LABEL):
        n_bad = np.sum(labels == UNKNOWN_LABEL)
        log.warning(
            f"[{subject_id}] {n_bad} epoch(s) have UNKNOWN_LABEL ({UNKNOWN_LABEL}). "
            "They will be kept in the file but should be filtered during training."
        )

    # ── 10. Shape into epochs ─────────────────────────────────────────────────
    x = signal.reshape(n_use, SAMPLES_EPOCH).astype(np.float32)   # (n_epochs, 3000)
    y = labels.astype(np.int32)                                    # (n_epochs,)
    assert x.shape[0] == y.shape[0], "Shape mismatch after alignment!"

    log.debug(f"[{subject_id}] x.shape={x.shape}, y.shape={y.shape}")

    # ── 11. Save .npz with audit metadata ─────────────────────────────────────
    out_filename = f"{subject_id}.npz"
    out_path     = os.path.join(OUTPUT_BASE_DIR, out_filename)

    np.savez(
        out_path,
        x          = x,
        y          = y,
        fs         = np.float32(TARGET_SFREQ),
        source_edf = np.bytes_(os.path.abspath(edf_path)),
        source_csv = np.bytes_(os.path.abspath(csv_path)),
    )
    log.info(f"[{subject_id}] ✓ Saved → {out_path}")

    result.update(status='OK', n_epochs=n_use, reason='')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("=" * 70)
    log.info("SHHS EEG PREPROCESSING PIPELINE")
    log.info(f"  BASE_DIR        : {BASE_DIR}")
    log.info(f"  OUTPUT_BASE_DIR : {OUTPUT_BASE_DIR}")
    log.info(f"  TARGET_SFREQ    : {TARGET_SFREQ} Hz")
    log.info(f"  EPOCH_SEC       : {EPOCH_SEC} s  ({SAMPLES_EPOCH} samples/epoch)")
    log.info(f"  MIN_DURATION    : {MIN_DURATION_MIN} min")
    log.info(f"  EEG_CHANNEL     : {EEG_CHANNEL}")
    log.info("=" * 70)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Collect subject directories
    if not os.path.isdir(BASE_DIR):
        log.error(f"BASE_DIR does not exist: {BASE_DIR}")
        sys.exit(1)

    subject_dirs = sorted([
        os.path.join(BASE_DIR, d)
        for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    log.info(f"Found {len(subject_dirs)} subject folder(s) in BASE_DIR.")

    # ── Iterate over subjects ─────────────────────────────────────────────────
    summary_rows = []

    for subject_dir in tqdm(subject_dirs, desc='Processing subjects', unit='subj'):
        # Skip if already done
        subject_id = os.path.basename(subject_dir)
        out_path   = os.path.join(OUTPUT_BASE_DIR, f"{subject_id}.npz")
        if os.path.isfile(out_path):
            log.info(f"[{subject_id}] Already processed. Skipping.")
            summary_rows.append(
                dict(subject=subject_id, status='SKIPPED', n_epochs='-', reason='already exists')
            )
            continue

        row = process_subject(subject_dir)
        summary_rows.append(row)

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 70)

    df_summary = pd.DataFrame(summary_rows)

    # Print aligned table to log
    col_widths = {
        'subject':  30,
        'status':   10,
        'n_epochs': 9,
        'reason':   45,
    }
    header = (
        f"{'Subject':<{col_widths['subject']}}"
        f"{'Status':<{col_widths['status']}}"
        f"{'Epochs':>{col_widths['n_epochs']}}"
        f"  {'Reason'}"
    )
    log.info(header)
    log.info("-" * 100)

    for _, row in df_summary.iterrows():
        line = (
            f"{str(row['subject']):<{col_widths['subject']}}"
            f"{str(row['status']):<{col_widths['status']}}"
            f"{str(row['n_epochs']):>{col_widths['n_epochs']}}"
            f"  {str(row['reason'])}"
        )
        log.info(line)

    log.info("-" * 100)
    counts = df_summary['status'].value_counts().to_dict()
    log.info(
        f"TOTAL: {len(df_summary)} | "
        + " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    )
    log.info(f"Log saved to: {_log_file}")

    # Save CSV summary alongside the output files
    summary_csv = os.path.join(OUTPUT_BASE_DIR, f'summary_{_log_ts}.csv')
    df_summary.to_csv(summary_csv, index=False)
    log.info(f"Summary CSV saved to: {summary_csv}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)
    run_pipeline()