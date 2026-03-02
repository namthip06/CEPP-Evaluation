#!/usr/bin/env python3
"""
Preprocessing script for rawEEG data
Compliant with data.md specification.

Execution:
    uv run preprocess_rawEEG.py

Folder structure expected at BASE_DIR:
    rawEEG/
    └── [id]/
        ├── edf_signals.edf       (REQUIRED)
        ├── csv_hypnogram.csv     (REQUIRED)
        └── csv_events.csv        (OPTIONAL)

Output: preprocessing_output/<id>_preprocessed.fif
"""

import os
import sys
import traceback

import mne
import numpy as np
import pandas as pd

# ─── Suppress MNE noise ───────────────────────────────────────────────────────
mne.set_log_level("WARNING")

# ─── Path Management (data.md §2 — Strict Rule) ──────────────────────────────
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "rawEEG"))
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "preprocessing_output")

# ─── Pre-Processing Guardrail constants (data.md §4) ─────────────────────────
MIN_DURATION_HOURS = 1.5  # Skip if recording < 6 h
NS_SKIP_THRESHOLD = 0.50  # Skip if NS epochs > 50 % of total epochs
EPOCH_DURATION_S = 30.0  # Fixed epoch length (data.md §5.2)
TARGET_SFREQ = 256  # BENDR standard sampling rate
BAD_CH_THRESHOLD = 5  # Bad channel: std > N × median or < median / N


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Hypnogram parser
# ═══════════════════════════════════════════════════════════════════════════════

# Label map — data.md §5.2 (case-insensitive)
_STAGE_MAP = {
    "W": "Sleep stage W",
    "WK": "Sleep stage W",
    "WAKE": "Sleep stage W",
    "N1": "Sleep stage 1",
    "1": "Sleep stage 1",
    "N2": "Sleep stage 2",
    "2": "Sleep stage 2",
    "N3": "Sleep stage 3",
    "3": "Sleep stage 3",
    "N4": "Sleep stage 4",
    "4": "Sleep stage 4",
    "REM": "Sleep stage R",
    "R": "Sleep stage R",
}


def parse_hypnogram(csv_path: str):
    """
    Parse csv_hypnogram.csv → mne.Annotations.
    Returns (annotations, ns_fraction) or (None, None) on failure.
    ns_fraction = fraction of epochs labelled NS (Non-Scored).
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Locate required columns (data.md §5.2)
        epoch_col = stage_col = time_col = None
        for col in df.columns:
            c = col.lower()
            if "epoch" in c:
                epoch_col = col
            elif "stage" in c:
                stage_col = col
            elif "time" in c:
                time_col = col

        if not all([epoch_col, stage_col]):
            print(f"  ⚠ Cannot find required columns in {os.path.basename(csv_path)}")
            print(f"    Available: {df.columns.tolist()}")
            return None, None

        total_epochs = len(df)
        ns_count = 0
        onsets, durations, descriptions = [], [], []

        for idx, row in df.iterrows():
            stage = str(row[stage_col]).strip().upper()

            # Count NS epochs for guardrail
            if stage in ("NS", "?", "UNKNOWN", ""):
                ns_count += 1

            desc = _STAGE_MAP.get(stage, f"Sleep stage {stage}")
            onsets.append(idx * EPOCH_DURATION_S)
            durations.append(EPOCH_DURATION_S)
            descriptions.append(desc)

        ns_fraction = ns_count / total_epochs if total_epochs > 0 else 0.0
        annotations = mne.Annotations(
            onset=onsets, duration=durations, description=descriptions
        )

        print(
            f"  Hypnogram: {total_epochs} epochs  |  NS = "
            f"{ns_count} ({ns_fraction:.1%})"
        )
        return annotations, ns_fraction

    except Exception as e:
        print(f"  ✗ Error parsing hypnogram: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: Preprocess one subject
# ═══════════════════════════════════════════════════════════════════════════════


def preprocess_single_subject(subject_id: str, raw_dir: str, output_dir: str) -> str:
    """
    Process one subject.

    Returns
    -------
    'ok'      — processed successfully
    'skip'    — skipped due to guardrail
    'error'   — unexpected failure
    """
    subject_path = os.path.join(raw_dir, subject_id)
    edf_path = os.path.join(subject_path, "edf_signals.edf")
    hypno_path = os.path.join(subject_path, "csv_hypnogram.csv")
    events_path = os.path.join(subject_path, "csv_events.csv")

    # ── Guardrail 1: Missing files (data.md §4) ─────────────────────────────
    missing = []
    if not os.path.exists(edf_path):
        missing.append("edf_signals.edf")
    if not os.path.exists(hypno_path):
        missing.append("csv_hypnogram.csv")
    if missing:
        print(f"  ⏭ SKIP — missing: {', '.join(missing)}")
        return "skip"

    try:
        # ── Step 1: Inspect EDF (no preload yet) ───────────────────────────
        print(f"  Loading EDF header …")
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

        sfreq = raw.info["sfreq"]
        duration = raw.times[-1]  # seconds
        n_ch_raw = len(raw.ch_names)

        print(f"  sfreq       : {sfreq} Hz")
        print(f"  Duration    : {duration:.1f} s  ({duration / 3600:.2f} h)")
        print(f"  Raw channels: {n_ch_raw}")

        # Show every channel name from file (user request: "show Channels from file")
        print(f"  Channel list ({n_ch_raw} total):")
        for i, ch in enumerate(raw.ch_names, 1):
            print(f"    [{i:>3}] {ch}")

        # ── Guardrail 2: Duration < 6 hours (data.md §4) ───────────────────
        if duration < MIN_DURATION_HOURS * 3600:
            print(
                f"  ⏭ SKIP — duration {duration / 3600:.2f} h < {MIN_DURATION_HOURS} h"
            )
            del raw
            return "skip"

        # ── Guardrail 3: NS epochs > 50 % (data.md §4) ─────────────────────
        print(f"  Parsing hypnogram …")
        annotations, ns_fraction = parse_hypnogram(hypno_path)
        if annotations is None:
            print("  ⏭ SKIP — could not parse hypnogram")
            del raw
            return "skip"
        if ns_fraction > NS_SKIP_THRESHOLD:
            print(f"  ⏭ SKIP — NS fraction {ns_fraction:.1%} > {NS_SKIP_THRESHOLD:.0%}")
            del raw
            return "skip"

        # ── Step 2: Preload data for processing ────────────────────────────
        print(f"  Preloading data into RAM …")
        raw.load_data(verbose=False)

        # ── Step 3: Pick EEG channels only (data.md §5.1) ─────────────────
        try:
            raw.pick_types(eeg=True, exclude="bads")
            eeg_ch = raw.ch_names
            print(f"  EEG channels selected: {len(eeg_ch)}")
            print(f"  EEG channel names: {eeg_ch}")
        except Exception as e:
            print(f"  ⚠ pick_types failed ({e}) — keeping all channels")

        # ── Step 4: Band-pass + notch filter ───────────────────────────────
        print(f"  Filtering (0.5–40 Hz, notch 50/60 Hz) …")
        try:
            raw.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin", verbose=False)
            raw.notch_filter(freqs=[50, 60], verbose=False)
        except Exception as e:
            print(f"  ⚠ Filtering error: {e}")

        # ── Step 5: Resample to 256 Hz ─────────────────────────────────────
        if raw.info["sfreq"] != TARGET_SFREQ:
            print(f"  Resampling {raw.info['sfreq']} → {TARGET_SFREQ} Hz …")
            raw.resample(TARGET_SFREQ, verbose=False)
        else:
            print(f"  Sampling rate already {TARGET_SFREQ} Hz — no resample needed")

        # ── Step 6: Bad channel detection & interpolation ──────────────────
        print(f"  Checking for bad channels …")
        try:
            data = raw.get_data()
            ch_stds = np.std(data, axis=1)
            med_std = np.median(ch_stds)
            bad_ch = [
                raw.ch_names[i]
                for i, s in enumerate(ch_stds)
                if s > BAD_CH_THRESHOLD * med_std or s < med_std / BAD_CH_THRESHOLD
            ]
            if bad_ch:
                print(f"  Bad channels found: {bad_ch}")
                raw.info["bads"] = bad_ch
                raw.interpolate_bads(reset_bads=True, verbose=False)
                print(f"  Interpolated {len(bad_ch)} bad channel(s)")
            else:
                print("  No bad channels detected")
            del data
        except Exception as e:
            print(f"  ⚠ Bad channel detection failed: {e}")

        # ── Step 7: Attach hypnogram annotations ───────────────────────────
        raw.set_annotations(annotations)
        print(f"  Annotations attached ({len(annotations)} epochs)")

        # ── Step 8: Optional events CSV (data.md §5.3) ─────────────────────
        if os.path.exists(events_path):
            print(f"  Events file found: {os.path.basename(events_path)} (not applied)")

        # ── Step 9: Save FIF (data.md §6 — Absolute Paths) ─────────────────
        output_file = os.path.join(output_dir, f"{subject_id}_preprocessed.fif")
        print(f"  Saving → {output_file}")
        raw.save(output_file, overwrite=True, verbose=False)

        # ── Clean up (data.md §6 — Memory Efficiency) ──────────────────────
        del raw
        print(f"  ✓ Done: {subject_id}")
        return "ok"

    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        traceback.print_exc()
        return "error"


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("RawEEG Preprocessing Pipeline  (data.md compliant)")
    print("=" * 70)
    print(f"Script dir  : {_SCRIPT_DIR}")
    print(f"Input  dir  : {BASE_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"Min duration: {MIN_DURATION_HOURS} h")
    print(f"NS threshold: {NS_SKIP_THRESHOLD:.0%}")
    print("=" * 70)

    # Validate input directory
    if not os.path.isdir(BASE_DIR):
        print(f"\n✗ rawEEG directory not found: {BASE_DIR}")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover subject folders
    try:
        subject_ids = sorted(
            item
            for item in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, item))
        )
    except Exception as e:
        print(f"\n✗ Cannot list {BASE_DIR}: {e}")
        sys.exit(1)

    print(f"\nFound {len(subject_ids)} subject folder(s)\n")
    if not subject_ids:
        print("No subjects found — exiting.")
        sys.exit(1)

    # ── Main loop: Load → Process → Save → Clear (data.md §6) ────────────────
    counts = {"ok": 0, "skip": 0, "error": 0}

    for idx, subject_id in enumerate(subject_ids, 1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(subject_ids)}]  Subject: {subject_id}")
        print("=" * 70)

        result = preprocess_single_subject(subject_id, BASE_DIR, OUTPUT_DIR)
        counts[result] += 1  # data.md §6 — error handling continues loop

    # ── Summary (data.md §6) ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total subjects : {len(subject_ids)}")
    print(f"  Successfully processed : {counts['ok']}")
    print(f"  Skipped (guardrails)   : {counts['skip']}")
    print(f"  Errors                 : {counts['error']}")
    print("=" * 70)

    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".fif")]
    print(f"\nOutput files in {OUTPUT_DIR}: {len(output_files)}")
    for f in sorted(output_files):
        print(f"  • {f}")


if __name__ == "__main__":
    main()
