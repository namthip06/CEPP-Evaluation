#!/usr/bin/env python3
"""
L-SeqSleepNet — End-to-End Inference Script
============================================
Pipeline:
  1. Preprocess raw EEG (EDF + CSV hypnogram) → .mat (HDF5) files
  2. Compute normalization parameters from the preprocessed data
  3. Run L-SeqSleepNet inference (TF1) using pretrained weights
  4. Save per-subject results to custom/results/[id]_test_ret.mat

Folder structure expected:
  /home/nummm/Documents/CEPP/rawEEG/[id]/
      csv_hypnogram.csv
      edf_signals.edf          ← skip subject if missing

Output:
  l-seqsleepnet/custom/preprocessing_output/
      [id]_eeg.mat             ← spectrogram .mat per subject
      test_list.txt            ← file list for DataGeneratorWrapper
  l-seqsleepnet/custom/results/
      [id]_test_ret.mat        ← yhat, score, acc per subject

Config: mirrors sleepedf-20/network/lseqsleepnet/config.py
        + pretrained SHHS weights (nsubseq=20, sub_seq_len=10)

Constraints:
  - No batch / no parallel processing (low RAM)
  - Absolute paths throughout
  - Skip subjects without edf_signals.edf
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
import hdf5storage
from scipy.signal import spectrogram as scipy_spectrogram

# ── Absolute paths ────────────────────────────────────────────────────────────
BASE_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_EEG_DIR     = os.path.abspath('/home/nummm/Documents/CEPP/rawEEG')
PREPROC_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), 'preprocessing_output'))
RESULTS_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))

# Pretrained model (SHHS, 1-channel, sub_seq_len=10, nsubseq=20, 1 block)
CHECKPOINT_DIR  = os.path.abspath(os.path.join(
    BASE_DIR, 'sleepedf-20',
    '__pretrained_shhs_1chan_subseqlen10_nsubseq20_1blocks'
))
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_model_acc')

# Network source directory (for imports)
NETWORK_DIR     = os.path.abspath(os.path.join(
    BASE_DIR, 'sleepedf-20', 'network', 'lseqsleepnet'
))

# ── Signal / spectrogram config (mirrors config.py) ───────────────────────────
SAMPLERATE    = 100        # Hz
EPOCH_SEC     = 30         # seconds per sleep epoch
EPOCH_SAMPLES = SAMPLERATE * EPOCH_SEC   # 3000
NFFT          = 256
FREQ_BINS     = 129        # nfft//2 + 1
TIME_FRAMES   = 29         # frame_seq_len
NPERSEG       = 256
NOVERLAP      = 148        # step=108 → 29 frames for 3000 samples
NCLASS        = 5

# ── Model hyper-parameters (pretrained SHHS checkpoint) ──────────────────────
SUB_SEQ_LEN   = 10
NSUBSEQ       = 20
SEQ_LEN       = SUB_SEQ_LEN * NSUBSEQ   # 200
NHIDDEN1      = 64
NHIDDEN2      = 64
NFILTER       = 32
ATTENTION_SIZE = 64
DUALRNN_BLOCKS = 1
DROPOUT_RNN   = 0.75
L2_REG_LAMBDA = 0.0001
LEARNING_RATE = 1e-4
FC_SIZE       = 512
LOWFREQ       = 0
HIGHFREQ      = 50
NCHANNEL      = 1

# Sleep stage mapping (0-indexed)
STAGE_MAP = {
    'WK': 0, 'W': 0, 'WAKE': 0,
    'N1': 1, '1': 1,
    'N2': 2, '2': 2,
    'N3': 3, '3': 3, 'N4': 3, '4': 3,
    'REM': 4, 'R': 4,
}


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_hypnogram(csv_path):
    """Parse csv_hypnogram.csv → list of 0-indexed integer labels."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        stage_col = next(
            (c for c in df.columns if 'stage' in c.lower()), None
        )
        if stage_col is None:
            print(f"  Warning: 'Sleep Stage' column not found. Columns: {df.columns.tolist()}")
            return None

        labels = []
        unknown = set()
        for _, row in df.iterrows():
            raw = str(row[stage_col]).strip().upper()
            lbl = STAGE_MAP.get(raw)
            if lbl is None:
                unknown.add(raw)
                lbl = 0
            labels.append(lbl)

        if unknown:
            print(f"  Warning: Unknown stages mapped to Wake: {unknown}")
        print(f"  Hypnogram: {len(labels)} epochs")
        return labels

    except Exception as e:
        print(f"  Error parsing hypnogram: {e}")
        return None


def compute_spectrogram(epoch_signal):
    """
    Compute log-power spectrogram for one 30-s epoch.

    Returns
    -------
    spec : np.ndarray, shape (TIME_FRAMES=29, FREQ_BINS=129)
    """
    _, _, Sxx = scipy_spectrogram(
        epoch_signal,
        fs=SAMPLERATE,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        window='hann',
        scaling='density',
        mode='psd',
    )
    # Sxx: (freq_bins, time_frames)
    n_freq = min(Sxx.shape[0], FREQ_BINS)
    n_time = min(Sxx.shape[1], TIME_FRAMES)
    spec = np.zeros((TIME_FRAMES, FREQ_BINS), dtype=np.float32)
    spec[:n_time, :n_freq] = np.log(Sxx[:n_freq, :n_time].T + 1e-10)
    return spec


def preprocess_subject(subject_id):
    """
    Preprocess one subject: EDF → spectrogram .mat file.

    Returns
    -------
    (output_path, n_epochs) on success, or (None, None) on skip/error.
    """
    subject_path   = os.path.join(RAW_EEG_DIR, subject_id)
    edf_path       = os.path.join(subject_path, 'edf_signals.edf')
    hypnogram_path = os.path.join(subject_path, 'csv_hypnogram.csv')
    output_path    = os.path.join(PREPROC_DIR, f'{subject_id}_eeg.mat')

    if not os.path.exists(edf_path):
        print(f"  Skip {subject_id}: edf_signals.edf not found")
        return None, None

    # Skip if already preprocessed
    if os.path.exists(output_path):
        # Read epoch count from existing file
        try:
            with h5py.File(output_path, 'r') as f:
                n_epochs = np.array(f['label']).shape[1]
            print(f"  Already preprocessed: {n_epochs} epochs")
            return output_path, n_epochs
        except Exception:
            pass  # re-process if file is corrupt

    print(f"\n  Preprocessing {subject_id}...")

    try:
        import mne
        mne.set_log_level('WARNING')

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        orig_sfreq = raw.info['sfreq']
        print(f"  sfreq={orig_sfreq}Hz | dur={raw.times[-1]:.0f}s | ch={len(raw.ch_names)}")

        # Pick EEG channels
        try:
            raw.pick_types(eeg=True, exclude='bads')
            if len(raw.ch_names) == 0:
                raise ValueError("No EEG channels")
        except Exception as e:
            print(f"  Warning: pick_types failed ({e}), using all channels")

        # Use only first channel (1-channel model)
        if len(raw.ch_names) > 1:
            raw.pick_channels([raw.ch_names[0]])

        # Resample to 100 Hz
        if orig_sfreq != SAMPLERATE:
            raw.resample(SAMPLERATE, verbose=False)

        signal = raw.get_data()[0]   # (total_samples,)
        del raw

        # Parse hypnogram
        if not os.path.exists(hypnogram_path):
            print(f"  Warning: hypnogram not found, skipping")
            del signal
            return None, None

        labels = parse_hypnogram(hypnogram_path)
        if labels is None:
            del signal
            return None, None

        # Align epochs
        max_sig_epochs = len(signal) // EPOCH_SAMPLES
        n_epochs = min(len(labels), max_sig_epochs)
        if n_epochs == 0:
            del signal
            return None, None
        labels = labels[:n_epochs]

        # Compute spectrograms
        print(f"  Computing {n_epochs} spectrograms...")
        X2    = np.zeros((TIME_FRAMES, FREQ_BINS, n_epochs), dtype=np.float32)
        y_hot = np.zeros((NCLASS, n_epochs), dtype=np.float32)
        label = np.zeros((1, n_epochs), dtype=np.float32)

        for ep in range(n_epochs):
            s = ep * EPOCH_SAMPLES
            spec = compute_spectrogram(signal[s:s + EPOCH_SAMPLES].astype(np.float64))
            X2[:, :, ep] = spec
            lbl = labels[ep]
            label[0, ep] = lbl
            y_hot[lbl, ep] = 1.0

        del signal

        # Save .mat (HDF5)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('X2',    data=X2,    compression='gzip', compression_opts=4)
            f.create_dataset('y',     data=y_hot, compression='gzip', compression_opts=4)
            f.create_dataset('label', data=label, compression='gzip', compression_opts=4)

        del X2, y_hot, label
        print(f"  Saved: {output_path}  ({n_epochs} epochs)")
        return output_path, n_epochs

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — NORMALIZATION (computed from test data itself, per-subject)
# ═════════════════════════════════════════════════════════════════════════════

def compute_norm_params(mat_path):
    """
    Compute mean and std of X2 from a single .mat file.
    Uses incremental formula to avoid loading full array twice.

    Returns
    -------
    meanX : np.ndarray, shape (FREQ_BINS,)
    stdX  : np.ndarray, shape (FREQ_BINS,)
    """
    with h5py.File(mat_path, 'r') as f:
        X2 = np.array(f['X2'], dtype=np.float32)
    # X2 stored as (TIME_FRAMES, FREQ_BINS, N) → transpose to (N, TIME_FRAMES, FREQ_BINS)
    X2 = np.transpose(X2, (2, 1, 0))
    N = X2.shape[0]
    X2_flat = X2.reshape(N * TIME_FRAMES, FREQ_BINS)
    meanX = X2_flat.mean(axis=0)
    stdX  = X2_flat.std(axis=0, ddof=1)
    stdX  = np.where(stdX < 1e-6, 1.0, stdX)   # avoid division by zero
    del X2, X2_flat
    return meanX, stdX


def normalize_X2(X2_transposed, meanX, stdX):
    """
    Normalize X2 in-place.

    Parameters
    ----------
    X2_transposed : np.ndarray, shape (N, TIME_FRAMES, FREQ_BINS)
    """
    N = X2_transposed.shape[0]
    flat = X2_transposed.reshape(N * TIME_FRAMES, FREQ_BINS)
    flat = (flat - meanX) / stdX
    return flat.reshape(N, TIME_FRAMES, FREQ_BINS)


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — INFERENCE (TF1 + L-SeqSleepNet)
# ═════════════════════════════════════════════════════════════════════════════

def build_config():
    """Build Config object matching the pretrained SHHS checkpoint."""
    sys.path.insert(0, NETWORK_DIR)
    from config import Config
    cfg = Config()
    cfg.sub_seq_len    = SUB_SEQ_LEN
    cfg.nsubseq        = NSUBSEQ
    cfg.nhidden1       = NHIDDEN1
    cfg.nhidden2       = NHIDDEN2
    cfg.nfilter        = NFILTER
    cfg.attention_size = ATTENTION_SIZE
    cfg.dualrnn_blocks = DUALRNN_BLOCKS
    cfg.dropout_rnn    = DROPOUT_RNN
    cfg.l2_reg_lambda  = L2_REG_LAMBDA
    cfg.learning_rate  = LEARNING_RATE
    cfg.fc_size        = FC_SIZE
    cfg.lowfreq        = LOWFREQ
    cfg.highfreq       = HIGHFREQ
    cfg.nchannel       = NCHANNEL
    cfg.samplerate     = SAMPLERATE
    cfg.nfft           = NFFT
    cfg.ndim           = FREQ_BINS
    cfg.frame_seq_len  = TIME_FRAMES
    cfg.nclass         = NCLASS
    return cfg


def run_inference_on_subject(subject_id, mat_path, n_epochs, sess, net, config):
    """
    Run L-SeqSleepNet inference on one subject's .mat file.

    Returns
    -------
    yhat  : np.ndarray, shape (N_valid, SEQ_LEN)   — predictions (1-indexed)
    score : np.ndarray, shape (N_valid, SEQ_LEN, 5) — softmax scores
    label : np.ndarray, shape (n_epochs,)            — ground-truth labels
    """
    # ── Load and normalize X2 ─────────────────────────────────────────────────
    with h5py.File(mat_path, 'r') as f:
        X2_raw = np.array(f['X2'], dtype=np.float32)   # (29, 129, N)
        lbl_raw = np.array(f['label'], dtype=np.float32)  # (1, N)
        y_raw   = np.array(f['y'], dtype=np.float32)      # (5, N)

    # Transpose to (N, TIME_FRAMES, FREQ_BINS)
    X2  = np.transpose(X2_raw, (2, 1, 0))   # (N, 29, 129)
    y   = np.transpose(y_raw,  (1, 0))       # (N, 5)
    lbl = np.squeeze(np.transpose(lbl_raw, (1, 0)))  # (N,)
    del X2_raw, y_raw, lbl_raw

    # Normalize using per-subject stats (no training data available)
    N = X2.shape[0]
    flat = X2.reshape(N * TIME_FRAMES, FREQ_BINS)
    meanX = flat.mean(axis=0)
    stdX  = flat.std(axis=0, ddof=1)
    stdX  = np.where(stdX < 1e-6, 1.0, stdX)
    flat  = (flat - meanX) / stdX
    X2    = flat.reshape(N, TIME_FRAMES, FREQ_BINS)
    del flat

    # Add channel dimension: (N, 29, 129, 1)
    X2 = np.expand_dims(X2, axis=-1)

    # ── Build valid index list (same logic as DataGenerator3) ─────────────────
    boundary = np.arange(0, SEQ_LEN - 1)   # first SEQ_LEN-1 epochs can't start a full window
    all_idx  = np.arange(N)
    mask     = np.in1d(all_idx, boundary, invert=True)
    data_index = all_idx[mask]

    if len(data_index) == 0:
        print(f"  Warning: Not enough epochs ({N}) for SEQ_LEN={SEQ_LEN}")
        return None, None, lbl

    n_valid = len(data_index)
    yhat  = np.zeros((n_valid, SEQ_LEN), dtype=np.float32)
    score = np.zeros((n_valid, SEQ_LEN, NCLASS), dtype=np.float32)

    # ── Inference: one sample at a time (no batch) ────────────────────────────
    print(f"  Running inference: {n_valid} windows...")
    for i, idx in enumerate(data_index):
        # Build sequence window: (1, SEQ_LEN, 29, 129, 1)
        x_seq = np.zeros((1, SEQ_LEN, TIME_FRAMES, FREQ_BINS, NCHANNEL), dtype=np.float32)
        y_seq = np.zeros((1, SEQ_LEN, NCLASS), dtype=np.float32)

        for t in range(SEQ_LEN):
            src = idx - (SEQ_LEN - 1) + t
            x_seq[0, t] = X2[src]
            y_seq[0, t] = y[src]

        # Reshape into (1, nsubseq, sub_seq_len, 29, 129, 1)
        x_in = np.zeros((1, NSUBSEQ, SUB_SEQ_LEN, TIME_FRAMES, FREQ_BINS, NCHANNEL), dtype=np.float32)
        y_in = np.zeros((1, NSUBSEQ, SUB_SEQ_LEN, NCLASS), dtype=np.float32)
        for s in range(NSUBSEQ):
            x_in[0, s] = x_seq[0, s * SUB_SEQ_LEN:(s + 1) * SUB_SEQ_LEN]
            y_in[0, s] = y_seq[0, s * SUB_SEQ_LEN:(s + 1) * SUB_SEQ_LEN]

        frame_seq_len_arr  = np.ones(SUB_SEQ_LEN * NSUBSEQ, dtype=int) * TIME_FRAMES
        sub_seq_len_arr    = np.ones(NSUBSEQ, dtype=int) * SUB_SEQ_LEN
        inter_subseq_len_arr = np.ones(SUB_SEQ_LEN, dtype=int) * NSUBSEQ

        feed = {
            net.input_x:          x_in,
            net.input_y:          y_in,
            net.dropout_rnn:      1.0,
            net.inter_subseq_len: inter_subseq_len_arr,
            net.sub_seq_len:      sub_seq_len_arr,
            net.frame_seq_len:    frame_seq_len_arr,
            net.istraining:       False,
        }

        _, _, yhat_i, score_i = sess.run(
            [net.output_loss, net.loss, net.prediction, net.score], feed
        )
        # yhat_i: (1, nsubseq, sub_seq_len) → flatten to (SEQ_LEN,)
        for s in range(NSUBSEQ):
            yhat[i, s * SUB_SEQ_LEN:(s + 1) * SUB_SEQ_LEN]  = yhat_i[0, s]
            score[i, s * SUB_SEQ_LEN:(s + 1) * SUB_SEQ_LEN] = score_i[0, s]

    yhat = yhat + 1   # make labels 1-indexed (W=1, N1=2, N2=3, N3=4, REM=5)
    del X2, y
    return yhat, score, lbl


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("L-SeqSleepNet — End-to-End Inference")
    print("=" * 70)
    print(f"Raw EEG    : {RAW_EEG_DIR}")
    print(f"Preproc    : {PREPROC_DIR}")
    print(f"Results    : {RESULTS_DIR}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print("=" * 70)

    os.makedirs(PREPROC_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.isdir(RAW_EEG_DIR):
        print(f"Error: rawEEG directory not found: {RAW_EEG_DIR}")
        sys.exit(1)

    if not os.path.exists(CHECKPOINT_PATH + '.index'):
        print(f"Error: Checkpoint not found: {CHECKPOINT_PATH}")
        print("  Expected files: best_model_acc.index / .meta / .data-*")
        sys.exit(1)

    # ── Get sorted subject list ───────────────────────────────────────────────
    subject_ids = sorted([
        d for d in os.listdir(RAW_EEG_DIR)
        if os.path.isdir(os.path.join(RAW_EEG_DIR, d))
    ])
    print(f"\nFound {len(subject_ids)} subject folders")

    # ── PHASE 1: Preprocess all subjects ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 1: Preprocessing")
    print("─" * 70)

    subject_mat_list = []   # [(subject_id, mat_path, n_epochs), ...]
    skip_count = 0

    for i, sid in enumerate(subject_ids, 1):
        print(f"\n[{i}/{len(subject_ids)}] {sid}")
        mat_path, n_epochs = preprocess_subject(sid)
        if mat_path is not None:
            subject_mat_list.append((sid, mat_path, n_epochs))
        else:
            skip_count += 1

    # Write test_list.txt
    test_list_path = os.path.join(PREPROC_DIR, 'test_list.txt')
    with open(test_list_path, 'w') as f:
        for sid, mat_path, n_epochs in subject_mat_list:
            f.write(f"{mat_path}\t{n_epochs}\n")

    print(f"\nPhase 1 complete: {len(subject_mat_list)} preprocessed, {skip_count} skipped")
    print(f"test_list.txt → {test_list_path}")

    if not subject_mat_list:
        print("No subjects to process. Exiting.")
        sys.exit(0)

    # ── PHASE 2: Load TF1 model (once) ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 2: Loading Model")
    print("─" * 70)

    sys.path.insert(0, NETWORK_DIR)
    import tensorflow as tf
    from lseqsleepnet import LSeqSleepNet

    config = build_config()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=gpu_options,
    )

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            net = LSeqSleepNet(config=config)

            # Build optimizer (required to restore all variables)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                optimizer   = tf.train.AdamOptimizer(config.learning_rate)
                grads_and_vars = optimizer.compute_gradients(net.loss)
                _ = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            saver = tf.train.Saver(tf.all_variables())
            saver.restore(sess, CHECKPOINT_PATH)
            print(f"Model loaded from: {CHECKPOINT_PATH}")

            # ── PHASE 3: Inference per subject ────────────────────────────────
            print("\n" + "─" * 70)
            print("PHASE 3: Inference")
            print("─" * 70)

            success_count = 0
            error_count   = 0

            for i, (sid, mat_path, n_epochs) in enumerate(subject_mat_list, 1):
                print(f"\n[{i}/{len(subject_mat_list)}] {sid}  ({n_epochs} epochs)")
                result_path = os.path.join(RESULTS_DIR, f'{sid}_test_ret.mat')

                # Skip if result already exists
                if os.path.exists(result_path):
                    print(f"  Already done, skipping.")
                    success_count += 1
                    continue

                try:
                    yhat, score, lbl = run_inference_on_subject(
                        sid, mat_path, n_epochs, sess, net, config
                    )

                    if yhat is None:
                        print(f"  Skipped: insufficient epochs")
                        error_count += 1
                        continue

                    # Save result
                    hdf5storage.savemat(
                        result_path,
                        {
                            'yhat':  yhat,
                            'score': score,
                            'label': lbl,
                        },
                        format='7.3',
                    )
                    print(f"  Saved → {result_path}")
                    success_count += 1

                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Preprocessed  : {len(subject_mat_list)}")
    print(f"Skipped (no EDF): {skip_count}")
    print(f"Inference OK  : {success_count}")
    print(f"Inference ERR : {error_count}")
    print(f"\nResults in: {RESULTS_DIR}")


if __name__ == '__main__':
    main()

# Preprocessed  : 65
# Skipped (no EDF): 0
# Inference OK  : 38
# Inference ERR : 27