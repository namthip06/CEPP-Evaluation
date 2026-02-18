#!/usr/bin/env python3
"""
Preprocessing script for L-SeqSleepNet
Converts raw EEG data from rawEEG/[id]/ to .mat (HDF5) files

Input folder structure:
    /home/nummm/Documents/CEPP/rawEEG/[id]/
    ├── csv_events.csv
    ├── csv_hypnogram.csv
    └── edf_signals.edf

Output:
    l-seqsleepnet/custom/preprocessing_output/
    ├── [id]_eeg.mat          # HDF5 mat file per subject
    └── test_list.txt         # file list for DataGeneratorWrapper

Expected .mat file format (read by datagenerator_from_list_v3.py via h5py):
    X2    : shape (29, 129, num_epochs)  -- time_frames x freq_bins x epochs
    y     : shape (5, num_epochs)         -- one-hot labels (0-indexed)
    label : shape (1, num_epochs)         -- integer labels (0-indexed)

Note: datagenerator_from_list_v3.py transposes with (2,1,0) after reading,
      so we store in MATLAB/HDF5 column-major order (reversed dims).

Sleep stage mapping (0-indexed, model adds +1 internally for display):
    WK / W / WAKE  -> 0
    N1 / 1         -> 1
    N2 / 2         -> 2
    N3 / 3 / N4/4  -> 3
    REM / R        -> 4
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
from scipy.signal import spectrogram as scipy_spectrogram

# ── Absolute paths ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_EEG_DIR = os.path.abspath('/home/nummm/Documents/CEPP/rawEEG')
OUTPUT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), 'preprocessing_output'))

# ── Config (mirrors sleepedf-20/network/lseqsleepnet/config.py) ──────────────
SAMPLERATE   = 100       # Hz  (config.samplerate)
NFFT         = 256       # config.nfft
EPOCH_SEC    = 30        # seconds per sleep epoch
EPOCH_SAMPLES = SAMPLERATE * EPOCH_SEC   # 3000 samples
FREQ_BINS    = 129       # config.ndim  (nfft//2 + 1)
TIME_FRAMES  = 29        # config.frame_seq_len
NCLASS       = 5         # number of sleep stages

# STFT parameters tuned to produce exactly 29 time frames per 30-s epoch
# With fs=100, epoch=3000 samples, nfft=256:
#   noverlap = nfft - (epoch_samples - 1) // (time_frames - 1)
#   We want ceil((epoch_samples - noverlap) / (nfft - noverlap)) = time_frames
# Empirically: nperseg=256, noverlap=148 → 29 frames for 3000 samples
NPERSEG   = 256
NOVERLAP  = 148   # step = 256 - 148 = 108

# Sleep stage label mapping (to 0-indexed integer)
STAGE_MAP = {
    'WK': 0, 'W': 0, 'WAKE': 0,
    'N1': 1, '1': 1,
    'N2': 2, '2': 2,
    'N3': 3, '3': 3, 'N4': 3, '4': 3,
    'REM': 4, 'R': 4,
}


# ─────────────────────────────────────────────────────────────────────────────
def parse_hypnogram(csv_path):
    """
    Parse csv_hypnogram.csv and return a list of integer labels (0-indexed).

    CSV format example:
        Epoch Number ,Start Time ,Sleep Stage
        1,9:18:51 PM,WK

    Returns
    -------
    labels : list[int] or None
        Integer label per epoch, or None on failure.
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Find the sleep stage column (flexible naming)
        stage_col = None
        for col in df.columns:
            if 'stage' in col.lower():
                stage_col = col
                break

        if stage_col is None:
            print(f"  Warning: 'Sleep Stage' column not found in {csv_path}")
            print(f"  Available columns: {df.columns.tolist()}")
            return None

        labels = []
        unknown_stages = set()
        for _, row in df.iterrows():
            stage_raw = str(row[stage_col]).strip().upper()
            label = STAGE_MAP.get(stage_raw, None)
            if label is None:
                unknown_stages.add(stage_raw)
                label = 0  # default to Wake for unknown stages
            labels.append(label)

        if unknown_stages:
            print(f"  Warning: Unknown sleep stages mapped to Wake: {unknown_stages}")

        print(f"  Parsed {len(labels)} epochs from hypnogram")
        return labels

    except Exception as e:
        print(f"  Error parsing hypnogram: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
def compute_spectrogram(epoch_signal):
    """
    Compute spectrogram for a single 30-second EEG epoch.

    Parameters
    ----------
    epoch_signal : np.ndarray, shape (EPOCH_SAMPLES,)

    Returns
    -------
    spec : np.ndarray, shape (TIME_FRAMES, FREQ_BINS)
        Log-power spectrogram clipped to [TIME_FRAMES, FREQ_BINS].
    """
    freqs, times, Sxx = scipy_spectrogram(
        epoch_signal,
        fs=SAMPLERATE,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        window='hann',
        scaling='density',
        mode='psd',
    )

    # Sxx shape: (freq_bins, time_frames) = (129, ~29)
    # Clip/pad to exact dimensions
    n_freq = min(Sxx.shape[0], FREQ_BINS)
    n_time = min(Sxx.shape[1], TIME_FRAMES)

    spec = np.zeros((TIME_FRAMES, FREQ_BINS), dtype=np.float32)
    spec[:n_time, :n_freq] = np.log(Sxx[:n_freq, :n_time].T + 1e-10)

    return spec  # shape: (TIME_FRAMES, FREQ_BINS) = (29, 129)


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_single_subject(subject_id, raw_dir, output_dir):
    """
    Preprocess one subject: load EDF, resample, epoch, compute spectrograms,
    save as HDF5 .mat file.

    Returns
    -------
    num_epochs : int or None
        Number of valid epochs saved, or None on failure/skip.
    """
    subject_path  = os.path.join(raw_dir, subject_id)
    edf_path       = os.path.join(subject_path, 'edf_signals.edf')
    hypnogram_path = os.path.join(subject_path, 'csv_hypnogram.csv')
    output_path    = os.path.join(output_dir, f'{subject_id}_eeg.mat')

    # ── Skip if EDF missing ───────────────────────────────────────────────────
    if not os.path.exists(edf_path):
        print(f"  Skipping {subject_id}: edf_signals.edf not found")
        return None

    print(f"\nProcessing: {subject_id}")

    try:
        # ── 1. Load EDF (lazy import to keep RAM low) ─────────────────────────
        import mne
        mne.set_log_level('WARNING')

        print("  Loading EDF...")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        orig_sfreq = raw.info['sfreq']
        print(f"  Original sfreq: {orig_sfreq} Hz | Duration: {raw.times[-1]:.1f}s | Channels: {len(raw.ch_names)}")

        # ── 2. Pick EEG channels only ─────────────────────────────────────────
        try:
            raw.pick_types(eeg=True, exclude='bads')
            if len(raw.ch_names) == 0:
                raise ValueError("No EEG channels found after pick_types")
            print(f"  EEG channels selected: {raw.ch_names}")
        except Exception as e:
            print(f"  Warning: pick_types failed ({e}), using all channels")

        # Use only the first channel (L-SeqSleepNet is 1-channel)
        if len(raw.ch_names) > 1:
            raw.pick_channels([raw.ch_names[0]])
            print(f"  Using channel: {raw.ch_names[0]}")

        # ── 3. Resample to 100 Hz ─────────────────────────────────────────────
        if orig_sfreq != SAMPLERATE:
            print(f"  Resampling {orig_sfreq} Hz → {SAMPLERATE} Hz...")
            raw.resample(SAMPLERATE, verbose=False)

        # ── 4. Get raw signal as 1D array ─────────────────────────────────────
        signal = raw.get_data()[0]  # shape: (total_samples,)
        del raw  # free RAM immediately

        total_samples = len(signal)
        print(f"  Signal samples: {total_samples} ({total_samples/SAMPLERATE:.1f}s)")

        # ── 5. Parse hypnogram ────────────────────────────────────────────────
        if not os.path.exists(hypnogram_path):
            print(f"  Warning: Hypnogram not found, skipping {subject_id}")
            del signal
            return None

        labels = parse_hypnogram(hypnogram_path)
        if labels is None:
            del signal
            return None

        # ── 6. Align epochs: use min(signal_epochs, hypnogram_epochs) ─────────
        max_signal_epochs = total_samples // EPOCH_SAMPLES
        n_epochs = min(len(labels), max_signal_epochs)

        if n_epochs == 0:
            print(f"  Warning: No valid epochs for {subject_id}")
            del signal
            return None

        labels = labels[:n_epochs]
        print(f"  Valid epochs: {n_epochs}")

        # ── 7. Compute spectrograms epoch by epoch ────────────────────────────
        print("  Computing spectrograms...")
        # Pre-allocate arrays
        X2    = np.zeros((TIME_FRAMES, FREQ_BINS, n_epochs), dtype=np.float32)
        y_hot = np.zeros((NCLASS, n_epochs), dtype=np.float32)
        label = np.zeros((1, n_epochs), dtype=np.float32)

        for ep_idx in range(n_epochs):
            start = ep_idx * EPOCH_SAMPLES
            end   = start + EPOCH_SAMPLES
            epoch_signal = signal[start:end].astype(np.float64)

            spec = compute_spectrogram(epoch_signal)  # (29, 129)
            X2[:, :, ep_idx] = spec  # store as (29, 129, N) for HDF5

            lbl = labels[ep_idx]
            label[0, ep_idx] = lbl
            y_hot[lbl, ep_idx] = 1.0

        del signal  # free RAM

        # ── 8. Save as HDF5 .mat ──────────────────────────────────────────────
        print(f"  Saving → {output_path}")
        with h5py.File(output_path, 'w') as f:
            # Store in MATLAB/HDF5 order: (29, 129, N), (5, N), (1, N)
            # datagenerator_from_list_v3.py reads and transposes (2,1,0):
            #   X2 becomes (N, 129, 29) → then used as (N, frame_seq_len, ndim)
            f.create_dataset('X2',    data=X2,    compression='gzip', compression_opts=4)
            f.create_dataset('y',     data=y_hot, compression='gzip', compression_opts=4)
            f.create_dataset('label', data=label, compression='gzip', compression_opts=4)

        del X2, y_hot, label

        print(f"  ✓ Done: {n_epochs} epochs saved")
        return n_epochs

    except Exception as e:
        print(f"  ✗ Error processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("L-SeqSleepNet Preprocessing Script")
    print("=" * 70)
    print(f"Input  : {RAW_EEG_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verify rawEEG directory exists
    if not os.path.isdir(RAW_EEG_DIR):
        print(f"\nError: rawEEG directory not found: {RAW_EEG_DIR}")
        sys.exit(1)

    # Get sorted list of subject IDs
    subject_ids = sorted([
        d for d in os.listdir(RAW_EEG_DIR)
        if os.path.isdir(os.path.join(RAW_EEG_DIR, d))
    ])
    print(f"\nFound {len(subject_ids)} subject folders")

    # ── Process each subject sequentially (no batch / no parallel) ───────────
    success_count = 0
    skip_count    = 0
    error_count   = 0
    file_list_entries = []  # for test_list.txt

    for i, subject_id in enumerate(subject_ids, 1):
        print(f"\n{'='*70}")
        print(f"Progress: {i}/{len(subject_ids)}  [{subject_id}]")
        print(f"{'='*70}")

        n_epochs = preprocess_single_subject(subject_id, RAW_EEG_DIR, OUTPUT_DIR)

        if n_epochs is None:
            edf_path = os.path.join(RAW_EEG_DIR, subject_id, 'edf_signals.edf')
            if not os.path.exists(edf_path):
                skip_count += 1
            else:
                error_count += 1
        else:
            success_count += 1
            mat_path = os.path.join(OUTPUT_DIR, f'{subject_id}_eeg.mat')
            file_list_entries.append(f"{mat_path}\t{n_epochs}")

    # ── Write test_list.txt ───────────────────────────────────────────────────
    test_list_path = os.path.join(OUTPUT_DIR, 'test_list.txt')
    with open(test_list_path, 'w') as f:
        for entry in file_list_entries:
            f.write(entry + '\n')
    print(f"\ntest_list.txt written: {test_list_path}")
    print(f"  Contains {len(file_list_entries)} subjects")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total subjects  : {len(subject_ids)}")
    print(f"Successfully processed : {success_count}")
    print(f"Skipped (no EDF)       : {skip_count}")
    print(f"Errors                 : {error_count}")
    print(f"\nOutput files in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
