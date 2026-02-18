#!/usr/bin/env python
# coding: utf-8

"""
Custom EDF/CSV Preprocessing Script for mulEEG
Based on SHHS preprocessing logic

Converts EDF signals and CSV hypnogram data into mulEEG-compatible .npz format

Folder structure expected:
    /home/nummm/Documents/CEPP/rawEEG/[id]/
        csv_events.csv
        csv_hypnogram.csv
        edf_signals.edf

Output saved to:
    custom/preprocessing_output/[id]/[id].npz
"""

import numpy as np
import os
import pandas as pd
from mne.io import read_raw_edf


# ==================== PATHS (absolute) ====================
# Base directory of this script (mulEEG/custom/)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Input: rawEEG folder (sibling of mulEEG)
BASE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', 'rawEEG'))

# Output: custom/preprocessing_output inside mulEEG
OUTPUT_BASE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, 'preprocessing_output'))
# ==========================================================


def parse_hypnogram_csv(csv_path, epoch_sec_size=30):
    """
    Parse hypnogram CSV file and convert sleep stages to numeric labels.

    Args:
        csv_path: Path to CSV file containing sleep stage labels
        epoch_sec_size: Duration of each epoch in seconds (default: 30)

    Returns:
        labels: numpy array of numeric sleep stage labels (0-4)

    Label mapping:
        0: Wake  (WK)
        1: N1
        2: N2
        3: N3 / N4
        4: REM

    Raises:
        ValueError: If an unknown sleep stage label is encountered
    """
    # Read CSV – strip leading/trailing whitespace from column names
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Locate the sleep stage column
    stage_col = None
    possible_cols = ['Sleep Stage', 'stage', 'label', 'hypnogram', 'annotation', 'event']
    for col in possible_cols:
        if col in df.columns:
            stage_col = col
            break

    if stage_col is None:
        stage_col = df.columns[0]
        print(f"  Warning: No standard stage column found, using first column '{stage_col}'")

    print(f"  Using column '{stage_col}' for sleep stages")

    stages = df[stage_col].values

    # Convert to numeric labels
    labels = []
    for stage in stages:
        stage_str = str(stage).strip().upper()

        if stage_str in ['W', 'WK', 'WAKE', '0']:
            labels.append(0)
        elif stage_str in ['N1', '1', 'S1']:
            labels.append(1)
        elif stage_str in ['N2', '2', 'S2']:
            labels.append(2)
        elif stage_str in ['N3', 'N4', '3', '4', 'S3', 'S4']:
            labels.append(3)
        elif stage_str in ['REM', 'R', '5']:
            labels.append(4)
        else:
            raise ValueError(
                f"Unknown sleep stage label '{stage}' in {csv_path}. "
                f"Expected one of: WK, N1, N2, N3, N4, REM (and numeric equivalents)."
            )

    return np.array(labels, dtype=np.int32)


def preprocess_edf_csv(edf_path, hypnogram_path, output_path,
                       select_channel=None, trim_wake_edges=True,
                       edge_minutes=30, epoch_sec_size=30):
    """
    Preprocess EDF and CSV files into mulEEG-compatible .npz format.

    Args:
        edf_path: Path to EDF file
        hypnogram_path: Path to CSV hypnogram file
        output_path: Path to save output .npz file
        select_channel: Specific EEG channel to extract (None = auto-select first EEG)
        trim_wake_edges: Whether to trim wake periods at edges
        edge_minutes: Minutes to extend before/after sleep period
        epoch_sec_size: Epoch duration in seconds
    """
    print(f"  EDF:       {edf_path}")
    print(f"  Hypnogram: {hypnogram_path}")

    # ── Read EDF (no preload yet – memory optimisation) ──────────────────────
    raw = read_raw_edf(edf_path, preload=False, stim_channel=None, verbose=False)
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")
    print(f"  Channels ({len(raw.ch_names)}): {raw.ch_names}")

    # ── Channel selection ─────────────────────────────────────────────────────
    if select_channel is None:
        eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch.upper()]
        if eeg_channels:
            select_channel = eeg_channels[0]
            print(f"  Auto-selected EEG channel: {select_channel}")
        else:
            select_channel = raw.ch_names[0]
            print(f"  Warning: No EEG channel found, using first channel: {select_channel}")
    else:
        print(f"  Using specified channel: {select_channel}")

    # Pick only the needed channel BEFORE loading data (saves RAM)
    raw.pick_channels([select_channel])
    raw.load_data()

    # ── Resample to 100 Hz if needed ──────────────────────────────────────────
    original_sfreq = raw.info['sfreq']
    target_sfreq = 100.0
    if original_sfreq != target_sfreq:
        print(f"  Resampling {original_sfreq} Hz → {target_sfreq} Hz …")
        raw.resample(target_sfreq, npad='auto', verbose=False)
    else:
        print(f"  Sampling rate already {target_sfreq} Hz, no resampling needed")

    sampling_rate = raw.info['sfreq']

    # Get raw signal (shape: [n_samples])
    raw_ch = raw.get_data()[0]
    print(f"  Signal samples: {len(raw_ch)}")

    # ── Parse hypnogram ───────────────────────────────────────────────────────
    labels = parse_hypnogram_csv(hypnogram_path, epoch_sec_size)
    print(f"  Labels: {len(labels)}")

    # ── Align signal length with labels ───────────────────────────────────────
    samples_per_epoch = int(epoch_sec_size * sampling_rate)
    expected_samples = len(labels) * samples_per_epoch

    if len(raw_ch) != expected_samples:
        print(f"  Warning: Signal length ({len(raw_ch)}) ≠ expected ({expected_samples})")
        if len(raw_ch) > expected_samples:
            print(f"  Trimming signal to {expected_samples} samples")
            raw_ch = raw_ch[:expected_samples]
        else:
            print(f"  Padding signal to {expected_samples} samples")
            raw_ch = np.pad(raw_ch, (0, expected_samples - len(raw_ch)), mode='constant')

    # ── Split into epochs ─────────────────────────────────────────────────────
    n_epochs = len(labels)
    x = raw_ch.reshape(n_epochs, samples_per_epoch).astype(np.float32)
    y = labels.astype(np.int32)

    print(f"  Epochs shape: {x.shape}")
    dist = {0: 'WK', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    print("  Label distribution: " + ", ".join(
        f"{dist[k]}={np.sum(y == k)}" for k in range(5)))

    # ── Trim wake edges (SHHS-style) ──────────────────────────────────────────
    if trim_wake_edges:
        nw_idx = np.where(y != 0)[0]
        if len(nw_idx) > 0:
            start_idx = max(0, nw_idx[0] - (edge_minutes * 2))
            end_idx = min(len(y) - 1, nw_idx[-1] + (edge_minutes * 2))
            select_idx = np.arange(start_idx, end_idx + 1)
            print(f"  Before trim: {x.shape}")
            x = x[select_idx]
            y = y[select_idx]
            print(f"  After  trim: {x.shape}")
        else:
            print("  Warning: No non-wake epochs found; skipping trim")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(output_path, x=x, y=y, fs=sampling_rate)
    print(f"  Saved → {output_path}")


def main():
    # ==================== CONFIGURATION ====================
    # Input directory (rawEEG)
    input_base_dir = BASE_DIR

    # Output directory (custom/preprocessing_output)
    output_base_dir = OUTPUT_BASE_DIR

    # Channel selection (None = auto-select first EEG channel)
    # Example: "EEG C4-A1"
    select_channel = None

    # Wake-edge trimming
    trim_wake_edges = True
    edge_minutes = 30

    # Epoch size (seconds)
    epoch_sec_size = 30
    # =======================================================

    print("=" * 70)
    print("EDF/CSV Batch Preprocessing for mulEEG")
    print("=" * 70)
    print(f"Input  directory : {input_base_dir}")
    print(f"Output directory : {output_base_dir}")
    print("=" * 70)

    # Validate input directory
    if not os.path.exists(input_base_dir):
        print(f"ERROR: Input directory does not exist: {input_base_dir}")
        return

    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Collect and sort patient folders
    patient_folders = sorted(
        f for f in os.listdir(input_base_dir)
        if os.path.isdir(os.path.join(input_base_dir, f))
    )
    print(f"\nFound {len(patient_folders)} patient folder(s)\n")

    processed_count = 0
    skipped_no_edf = 0
    skipped_done = 0
    error_count = 0

    for patient_id in patient_folders:
        patient_input_dir = os.path.join(input_base_dir, patient_id)
        patient_output_dir = os.path.join(output_base_dir, patient_id)
        output_file = os.path.join(patient_output_dir, f"{patient_id}.npz")

        print(f"{'=' * 70}")
        print(f"Patient: {patient_id}")

        # Skip if already processed
        if os.path.exists(output_file):
            print(f"  ⊘ Already processed – skipping")
            skipped_done += 1
            continue

        # Required files
        edf_file = os.path.join(patient_input_dir, "edf_signals.edf")
        hypnogram_file = os.path.join(patient_input_dir, "csv_hypnogram.csv")

        # Skip if EDF missing
        if not os.path.exists(edf_file):
            print(f"  ⊘ edf_signals.edf not found – skipping")
            skipped_no_edf += 1
            continue

        # Error if hypnogram missing
        if not os.path.exists(hypnogram_file):
            print(f"  ✗ csv_hypnogram.csv not found – skipping with error")
            error_count += 1
            continue

        # Create output directory
        os.makedirs(patient_output_dir, exist_ok=True)

        try:
            preprocess_edf_csv(
                edf_path=edf_file,
                hypnogram_path=hypnogram_file,
                output_path=output_file,
                select_channel=select_channel,
                trim_wake_edges=trim_wake_edges,
                edge_minutes=edge_minutes,
                epoch_sec_size=epoch_sec_size,
            )
            processed_count += 1
            print(f"  ✓ Done")

        except ValueError as ve:
            # Label mapping error – raised intentionally
            print(f"  ✗ Label error: {ve}")
            error_count += 1
            # Clean up partial output
            if os.path.exists(patient_output_dir):
                import shutil
                shutil.rmtree(patient_output_dir)

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            error_count += 1
            if os.path.exists(patient_output_dir):
                import shutil
                shutil.rmtree(patient_output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total patient folders : {len(patient_folders)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (no EDF)      : {skipped_no_edf}")
    print(f"Skipped (already done): {skipped_done}")
    print(f"Errors                : {error_count}")
    print("=" * 70)


if __name__ == "__main__":
    seed = 123
    np.random.seed(seed)
    main()
