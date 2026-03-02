#!/home/nummm/Documents/CEPP/BENDR/.venv/bin/python
"""
BENDR Sequence Prediction Pipeline (Custom - No Config Required)
================================================================
Reads preprocessed FIF files directly from custom/preprocessing_output,
runs BENDR Encoder + Contextualizer via BendingCollegeWav2Vec, and
saves evaluation results as both CSV and Excel.

Usage:
    python custom/sequence_prediction.py

No YAML or argparse needed. All settings are hardcoded below.
"""

import os
import sys

# ─── Venv guard: re-exec with the correct interpreter if needed ───────────────
_VENV_PYTHON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "python")
)
if os.path.exists(_VENV_PYTHON) and os.path.realpath(
    sys.executable
) != os.path.realpath(_VENV_PYTHON):
    import subprocess

    result = subprocess.run([_VENV_PYTHON] + sys.argv)
    sys.exit(result.returncode)
# ─────────────────────────────────────────────────────────────────────────────

import time
import csv
import datetime
import traceback

import mne
import tqdm
import psutil
import numpy as np
import pandas as pd
import torch

# ─── Suppress MNE noise ──────────────────────────────────────────────────────
mne.set_log_level(False)

# ─── Add BENDR root to sys.path so local imports work ────────────────────────
BENDR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BENDR_ROOT not in sys.path:
    sys.path.insert(0, BENDR_ROOT)

from dn3_ext import ConvEncoderBENDR, BENDRContextualizer, BendingCollegeWav2Vec

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — IN-CODE CONFIGURATION  (edit here instead of any .yml file)
# ═══════════════════════════════════════════════════════════════════════════════

# --- Paths ---
PREPROCESSING_OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "preprocessing_output")
)
WEIGHTS_DIR = os.path.join(BENDR_ROOT, "weights")
ENCODER_WEIGHTS = os.path.join(WEIGHTS_DIR, "encoder.pt")
CONTEXT_WEIGHTS = os.path.join(WEIGHTS_DIR, "contextualizer.pt")
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
RESULTS_EXCEL = os.path.join(OUTPUT_DIR, "seq_results.xlsx")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "seq_results.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")

# --- Model hyper-parameters ---
SAMPLING_RATE = 256  # Hz — BENDR standard
SEQUENCE_LENGTH = 15360  # samples = 60 seconds @ 256 Hz
HIDDEN_SIZE = 512  # ConvEncoderBENDR encoder_h
LAYER_DROP = 0.01  # BENDRContextualizer layer_drop

# --- Training ---
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 1  # increase as needed
NUM_WORKERS = 0  # set >0 if your system supports it

# --- Sequence-length sweep (set both to None to skip sweep) ---
MIN_SEQUENCE = None  # e.g. 1024 to enable sweep
NUM_SEQUENCE = None  # number of log-spaced points

# --- GPU flag ---
# Set to True to force CPU (e.g. when cuDNN version is incompatible).
# The installed PyTorch expects cuDNN 9.10.x which may not be present.
FORCE_CPU = True

# --- EEG channels to extract from PSG FIF files ---
# The FIF files are full Polysomnography recordings that contain mixed signals
# (EMG, ECG, respiration, etc.).  BENDR was designed for EEG only, so we
# select just the real EEG derivations.  Channels not found in a particular
# file are silently skipped; the encoder input is then zero-padded to 20 ch.
EEG_CHANNELS = [
    "EEG F3-A2",
    "EEG F4-A1",
    "EEG A1-A2",
    "EEG C3-A2",
    "EEG C4-A1",
    "EEG O1-A2",
    "EEG O2-A1",
]

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — LOGGING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _log(msg: str):
    """Timestamped console print that plays well with tqdm."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    tqdm.tqdm.write(f"[{timestamp}] {msg}")


def _mem_info() -> str:
    """Returns a short string with RAM and (optionally) VRAM usage."""
    ram_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
    info = f"RAM={ram_gb:.2f} GB"
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1e6
        info += f"  VRAM={vram_mb:.0f} MB"
    return info


def _init_log_csv(path: str):
    """Create (or reset) the per-epoch CSV log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject_id",
                "dataset",
                "seq_len",
                "loss",
                "accuracy",
                "mask_pct",
                "elapsed_s",
                "samples_per_sec",
            ]
        )
    _log(f"Log file initialised → {path}")


def _append_log_csv(path: str, row: dict):
    """Append one result row to the CSV log."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "dataset",
                "seq_len",
                "loss",
                "accuracy",
                "mask_pct",
                "elapsed_s",
                "samples_per_sec",
            ],
        )
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — DATA INTEGRATION  (load FIF files without YAML)
# ═══════════════════════════════════════════════════════════════════════════════


def list_fif_files(directory: str) -> list:
    """Return sorted list of absolute paths to all .fif files in *directory*."""
    files = sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".fif") or f.endswith("_preprocessed.fif")
        ]
    )
    return files


def load_raw_fif(fif_path: str):
    """
    Load a preprocessed FIF file and return an MNE Raw object with **only**
    the EEG channels listed in EEG_CHANNELS.

    Steps
    -----
    1. Read the raw FIF (all channels).
    2. Intersect EEG_CHANNELS with the channels actually present in the file.
    3. Keep only those channels (ordered as defined in EEG_CHANNELS).
    4. Resample to SAMPLING_RATE if necessary.
    """
    subject_id = (
        os.path.basename(fif_path).replace("_preprocessed.fif", "").replace(".fif", "")
    )
    _log(f"  Loading FIF: {os.path.basename(fif_path)}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    _log(
        f"  sfreq={raw.info['sfreq']} Hz  total_channels={len(raw.ch_names)}"
        f"  duration={raw.times[-1]:.1f}s"
    )

    # ── Select EEG-only channels ──────────────────────────────────────────────
    available = set(raw.ch_names)
    selected = [ch for ch in EEG_CHANNELS if ch in available]
    missing = [ch for ch in EEG_CHANNELS if ch not in available]

    if missing:
        _log(f"  ⚠ EEG channels not found in file (will be zero-padded): {missing}")
    if not selected:
        raise RuntimeError(
            f"None of the expected EEG channels were found in {fif_path}.\n"
            f"File contains: {raw.ch_names}"
        )

    raw.pick_channels(selected, ordered=True)
    _log(f"  ✓ Kept {len(selected)}/{len(EEG_CHANNELS)} EEG channels: {selected}")

    # ── Resample if needed ────────────────────────────────────────────────────
    if raw.info["sfreq"] != SAMPLING_RATE:
        _log(f"  Resampling {raw.info['sfreq']} → {SAMPLING_RATE} Hz …")
        raw.resample(SAMPLING_RATE, verbose=False)

    return subject_id, raw


def raw_to_tensor(raw) -> torch.Tensor:
    """
    Convert MNE Raw to a (1, channels, samples) float32 Tensor.
    Crops / pads to SEQUENCE_LENGTH and applies To1020 normalisation.
    """
    data = raw.get_data().astype(np.float32)  # (channels, total_samples)
    n_ch, n_samp = data.shape

    # Crop or pad to SEQUENCE_LENGTH
    if n_samp >= SEQUENCE_LENGTH:
        data = data[:, :SEQUENCE_LENGTH]
    else:
        pad = np.zeros((n_ch, SEQUENCE_LENGTH - n_samp), dtype=np.float32)
        data = np.concatenate([data, pad], axis=1)

    tensor = torch.from_numpy(data).unsqueeze(0)  # → (1, C, T)
    return tensor


def adapt_channels_to_20(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adjust the channel dimension to exactly 20 so that ConvEncoderBENDR(20, …)
    receives the correct shape.

    * If the tensor already has ≥ 20 channels, the first 20 are kept.
    * If it has fewer (typical case: 7 EEG channels), zero-channels are appended
      so the total reaches 20.  The zero channels carry no information but allow
      the pre-trained encoder weights to load without modification.
    """
    _, C, T = tensor.shape
    target = 20
    if C >= target:
        if C > target:
            _log(f"  adapt_channels_to_20: truncating {C} → {target} channels")
        return tensor[:, :target, :]
    else:
        n_pad = target - C
        _log(
            f"  adapt_channels_to_20: zero-padding {C} → {target} channels "
            f"({n_pad} dummy channel(s) appended)"
        )
        pad = torch.zeros(1, n_pad, T, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — TRAINING & EVALUATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════


def build_process(seq_len: int):
    """
    Instantiate encoder + contextualizer + BendingCollegeWav2Vec process.
    Loads pretrained weights; no YAML required.
    """
    _log(f"Building encoder  (hidden={HIDDEN_SIZE}) …")
    encoder = ConvEncoderBENDR(20, encoder_h=HIDDEN_SIZE)

    _log(f"Building contextualizer  (layer_drop={LAYER_DROP}) …")
    contextualizer = BENDRContextualizer(encoder.encoder_h, layer_drop=LAYER_DROP)

    _log(f"Loading encoder weights  ← {ENCODER_WEIGHTS}")
    if os.path.exists(ENCODER_WEIGHTS):
        encoder.load(ENCODER_WEIGHTS)
    else:
        _log("  ⚠ Encoder weights NOT found — using random init")

    _log(f"Loading contextualizer weights  ← {CONTEXT_WEIGHTS}")
    if os.path.exists(CONTEXT_WEIGHTS):
        contextualizer.load(CONTEXT_WEIGHTS)
    else:
        _log("  ⚠ Contextualizer weights NOT found — using random init")

    # Model description
    desc = encoder.description(SAMPLING_RATE, seq_len)
    _log(f"Encoder: {desc}")

    process = BendingCollegeWav2Vec(
        encoder,
        contextualizer,
        learning_rate=LEARNING_RATE,
        cuda="cpu" if FORCE_CPU else None,  # None = auto-detect GPU
    )
    return process


def evaluate_subject(
    process,
    subject_id: str,
    tensor: torch.Tensor,
    seq_len: int,
    dataset_name: str = "custom",
) -> dict:
    """
    Run a single forward pass (eval mode) and return a metrics dict.
    BaseProcess stores the active device as process.device (not nn.Module).
    """
    device = process.device  # BaseProcess always exposes .device
    tensor = tensor.to(device)

    process.train(False)  # BaseProcess uses .train(bool), not .eval()
    t0 = time.time()
    with torch.no_grad():
        try:
            outputs = process.forward(tensor)
        except Exception as e:
            _log(f"  ✗ Forward pass failed for {subject_id}: {e}")
            return None

    elapsed = time.time() - t0
    sps = SEQUENCE_LENGTH / elapsed if elapsed > 0 else float("nan")

    # Extract metrics from outputs  (logits, encoded_z, mask)
    logits, z, mask = outputs
    labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
    loss_val = torch.nn.CrossEntropyLoss()(logits, labels).item()
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    mask_pct = mask.float().mean().item()

    return {
        "subject_id": subject_id,
        "dataset": dataset_name,
        "seq_len": seq_len,
        "loss": round(loss_val, 6),
        "accuracy": round(acc, 4),
        "mask_pct": round(mask_pct, 4),
        "elapsed_s": round(elapsed, 3),
        "samples_per_sec": round(sps, 1),
    }


def run_pipeline(fif_files: list, seq_len: int, all_results: list):
    """
    Build process once, iterate over all FIF files, evaluate each subject.
    """
    _log("=" * 65)
    _log(f"Sequence length: {seq_len} samples  ({seq_len / SAMPLING_RATE:.1f} s)")
    _log(f"Memory before build: {_mem_info()}")

    process = build_process(seq_len)
    process.train(False)  # BaseProcess uses .train(bool), not .eval()

    _log(f"Memory after build: {_mem_info()}")
    _log(f"Processing {len(fif_files)} subject(s) …")

    for fif_path in tqdm.tqdm(fif_files, desc="Subjects", unit="subj"):
        subject_id, raw = load_raw_fif(fif_path)
        _log("  Converting to tensor ...")
        tensor = raw_to_tensor(raw)
        tensor = adapt_channels_to_20(tensor)
        del raw  # free RAM

        _log(
            f"  Evaluating {subject_id} | tensor={tuple(tensor.shape)} | {_mem_info()}"
        )
        metrics = evaluate_subject(process, subject_id, tensor, seq_len)
        if metrics is None:
            continue

        all_results.append(metrics)
        _append_log_csv(LOG_FILE, metrics)

        _log(
            f"  ✓ {subject_id}  loss={metrics['loss']:.4f}  "
            f"acc={metrics['accuracy']:.4f}  "
            f"mask%={metrics['mask_pct']:.4f}  "
            f"{metrics['samples_per_sec']:.0f} samp/s"
        )

    # Auto-checkpoint: save full process state after each seq_len sweep
    if all_results:
        best = max(all_results, key=lambda x: x["accuracy"])
        _log(f"Best so far → {best['subject_id']}  acc={best['accuracy']}")
        try:
            # BaseProcess.save_best() returns list of state_dicts for each trainable module
            checkpoint = process.save_best()
            torch.save(checkpoint, BEST_MODEL_PATH)
            _log(f"  💾 Checkpoint saved → {BEST_MODEL_PATH}")
        except Exception as e:
            _log(f"  ⚠ Could not save checkpoint: {e}")


def save_results(all_results: list):
    """Save aggregated results to both CSV and Excel."""
    if not all_results:
        _log("No results to save.")
        return

    df = pd.DataFrame(all_results)

    # CSV
    df.to_csv(RESULTS_CSV, index=False)
    _log(f"Results CSV  → {RESULTS_CSV}")

    # Excel
    try:
        with pd.ExcelWriter(RESULTS_EXCEL, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="sequence_results", index=False)
        _log(f"Results Excel → {RESULTS_EXCEL}")
    except Exception as e:
        _log(f"⚠ Excel export failed: {e}")

    # Summary to console
    _log("─" * 65)
    _log("Summary statistics:")
    tqdm.tqdm.write(
        df[["loss", "accuracy", "mask_pct", "samples_per_sec"]].describe().to_string()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    _log("=" * 65)
    _log("BENDR Sequence Prediction — Custom Pipeline (No Config)")
    _log("=" * 65)
    _log(f"Preprocessing dir : {PREPROCESSING_OUTPUT_DIR}")
    _log(f"Encoder weights   : {ENCODER_WEIGHTS}")
    _log(f"Context weights   : {CONTEXT_WEIGHTS}")
    _log(f"Output dir        : {OUTPUT_DIR}")
    _log(f"Sequence length   : {SEQUENCE_LENGTH} samples @ {SAMPLING_RATE} Hz")
    _log(f"Hidden size       : {HIDDEN_SIZE}  |  Layer drop: {LAYER_DROP}")
    _log(f"System RAM total  : {psutil.virtual_memory().total / 1e9:.1f} GB")
    if torch.cuda.is_available():
        _log(f"GPU               : {torch.cuda.get_device_name(0)}")
    else:
        _log("GPU               : not available (CPU mode)")
    _log("=" * 65)

    # Prepare output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _init_log_csv(LOG_FILE)

    # Discover FIF files
    _log(f"Scanning for FIF files in: {PREPROCESSING_OUTPUT_DIR}")
    if not os.path.isdir(PREPROCESSING_OUTPUT_DIR):
        _log(f"✗ Preprocessing output directory not found: {PREPROCESSING_OUTPUT_DIR}")
        sys.exit(1)

    fif_files = list_fif_files(PREPROCESSING_OUTPUT_DIR)
    _log(f"Detected {len(fif_files)} FIF file(s):")
    for f in fif_files:
        _log(f"  • {os.path.basename(f)}")

    if not fif_files:
        _log("No FIF files found. Run preprocess_rawEEG.py first.")
        sys.exit(1)

    all_results = []
    script_start = time.time()

    # ─── Sequence-length sweep (optional) ─────────────────────────────────────
    if MIN_SEQUENCE is not None and NUM_SEQUENCE is not None:
        logspace = list(
            reversed(
                np.logspace(
                    np.log10(MIN_SEQUENCE), np.log10(SEQUENCE_LENGTH), num=NUM_SEQUENCE
                ).astype(int)
            )
        )
        _log(
            f"Sequence sweep mode: {NUM_SEQUENCE} points from {MIN_SEQUENCE} to {SEQUENCE_LENGTH}"
        )
        for seq_len in tqdm.tqdm(logspace, desc="Sequence lengths"):
            tqdm.tqdm.write(f"\n{'═' * 65}")
            tqdm.tqdm.write(f"Processing sequence length: {seq_len}")
            run_pipeline(fif_files, seq_len, all_results)
    else:
        # Single run at full SEQUENCE_LENGTH
        run_pipeline(fif_files, SEQUENCE_LENGTH, all_results)

    # ─── Save all results ──────────────────────────────────────────────────────
    save_results(all_results)

    total_time = time.time() - script_start
    _log("=" * 65)
    _log(f"Pipeline complete in {total_time:.1f}s  ({total_time / 60:.1f} min)")
    _log(f"Final memory: {_mem_info()}")
    _log("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _log("\n⚠ Interrupted by user.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
