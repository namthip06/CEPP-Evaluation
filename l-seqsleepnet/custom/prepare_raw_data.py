import os
import gc
import logging
import mne
import numpy as np
import pandas as pd
from scipy.io import savemat

# Set up logging configuration for debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define absolute paths
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "rawEEG"))
OUTPUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "processed_data"))


def extract_events_from_csv(csv_path):
    """
    Extracts events from csv_events.csv for feature engineering.
    """
    logging.debug(f"Extracting events from {csv_path}")
    try:
        events_df = pd.read_csv(csv_path)
        return events_df
    except Exception as e:
        logging.error(f"Failed to read events CSV: {e}")
        return None


def normalize_sleep_stage(label):
    if pd.isna(label):
        return "NS"
    raw_label = str(label).strip().upper()

    if raw_label in ["W", "WK", "WAKE"]:
        return "W"
    elif raw_label in ["N1", "1"]:
        return "N1"
    elif raw_label in ["N2", "2"]:
        return "N2"
    elif raw_label in ["N3", "3"]:
        return "N3"
    elif raw_label in ["REM", "R"]:
        return "R"
    else:
        return "NS"


def process_subject(subject_id, subject_dir):
    logging.debug(f"--- Processing subject: {subject_id} ---")

    edf_signals_path = os.path.join(subject_dir, "edf_signals.edf")
    csv_hypnogram_path = os.path.join(subject_dir, "csv_hypnogram.csv")
    csv_events_path = os.path.join(subject_dir, "csv_events.csv")

    # Pre-Processing Guardrails (Skip Rules)

    # 1. Missing Files
    if not os.path.exists(edf_signals_path):
        logging.warning(f"Skipping {subject_id}: Missing {edf_signals_path}")
        return False

    if not os.path.exists(csv_hypnogram_path):
        logging.warning(f"Skipping {subject_id}: Missing {csv_hypnogram_path}")
        return False

    # Component Specifications -> 5.1 Raw Signals (edf_signals.edf)
    logging.debug(f"Loading raw EDF for {subject_id}...")
    try:
        raw = mne.io.read_raw_edf(edf_signals_path, preload=True, verbose=False)
    except Exception as e:
        logging.error(f"Skipping {subject_id}: Failed to load EDF. Error: {e}")
        return False

    # 2. Recording Duration
    total_duration_sec = raw.times[-1]
    total_hours = total_duration_sec / 3600.0
    if total_hours < 1.5:
        logging.warning(
            f"Skipping {subject_id}: Recording duration is {total_hours:.2f}h (< 1.5 hours)."
        )
        return False
    logging.debug(f"Recording duration for {subject_id} is {total_hours:.2f}h.")

    # Bandpass Filter (0.3 - 40 Hz)
    logging.debug(f"Filtering {subject_id}: Bandpass 0.3 - 40 Hz")
    raw.filter(l_freq=0.3, h_freq=40.0, fir_design="firwin", verbose=False)

    # Integrity: Ensure the sampling frequency is consistent across the recording
    fs = raw.info["sfreq"]
    if fs <= 0:
        logging.warning(f"Skipping {subject_id}: Invalid sampling frequency '{fs}'.")
        return False

    logging.debug(f"Original sampling frequency for {subject_id} is {fs} Hz.")

    # Resample if fs > 100 Hz
    if fs > 100.0:
        logging.debug(f"Resampling {subject_id} from {fs}Hz to 100.0Hz...")
        raw.resample(100.0, npad="auto")
        fs = 100.0
        logging.debug(f"Resampling complete. New sfreq is {fs}Hz.")

    # Channel Selection: Select specific EEG and EOG channels
    eeg_candidates = ["EEG C3-A2", "EEG C4-A1", "C3-A2", "C4-A1"]
    eog_candidates = ["EOG LOC-A2", "EOG ROC-A2", "LOC-A2", "ROC-A2"]

    selected_channels = []

    # Pick first matching EEG channel
    for ch in eeg_candidates:
        if ch in raw.ch_names:
            selected_channels.append(ch)
            break

    # Pick first matching EOG channel
    for ch in eog_candidates:
        if ch in raw.ch_names:
            selected_channels.append(ch)
            break

    if len(selected_channels) < 2:
        logging.warning(
            f"Skipping {subject_id}: Required EEG/EOG channels not found. Available: {raw.ch_names}"
        )
        return False

    logging.debug(f"Retaining channels: {selected_channels}")
    raw.pick(selected_channels)

    # Component Specifications -> 5.2 Hypnogram (csv_hypnogram.csv)
    logging.debug(f"Loading hypnogram for {subject_id}...")
    try:
        hypno_df = pd.read_csv(csv_hypnogram_path)
    except Exception as e:
        logging.error(f"Skipping {subject_id}: Failed to load hypnogram. Error: {e}")
        return False

    required_hypno_cols = ["Epoch Number", "Start Time", "Sleep Stage"]

    # Strip whitespace from column names to handle inconsistencies
    hypno_df.columns = hypno_df.columns.str.strip()

    if not all(col in hypno_df.columns for col in required_hypno_cols):
        logging.warning(
            f"Skipping {subject_id}: Missing one or more required hypnogram columns {required_hypno_cols}."
        )
        return False

    # Label Normalization
    hypno_df["Normalized_Stage"] = hypno_df["Sleep Stage"].apply(normalize_sleep_stage)

    # 3. Data Quality (NS)
    total_epochs = len(hypno_df)
    ns_epochs = (hypno_df["Normalized_Stage"] == "NS").sum()

    if total_epochs == 0:
        logging.warning(f"Skipping {subject_id}: Hypnogram contains 0 epochs.")
        return False

    if (ns_epochs / total_epochs) > 0.5:
        logging.warning(
            f"Skipping {subject_id}: Exceeds 50% NS epochs ({ns_epochs}/{total_epochs})."
        )
        return False

    logging.debug(
        f"{subject_id} Data Quality: {ns_epochs}/{total_epochs} NS epochs ({(ns_epochs / total_epochs) * 100:.2f}%)."
    )

    # Epoch length is fixed at 30.0 seconds
    logging.debug(
        "Verified hypnogram metadata, standard epoch length assumed to be 30.0s."
    )

    # Component Specifications -> 5.3 Events (csv_events.csv)
    if os.path.exists(csv_events_path):
        logging.debug(f"Events file found for {subject_id}. Extracting events...")
        events_df = extract_events_from_csv(csv_events_path)
        if events_df is not None:
            logging.debug(f"Extracted {len(events_df)} events for {subject_id}.")
    else:
        logging.debug(
            f"No events file found for {subject_id}, skipping event extraction."
        )

    # Apply Filtering
    logging.debug("Applying bandpass filter 0.3 - 40.0 Hz...")
    raw.filter(l_freq=0.3, h_freq=40.0, verbose=False)

    # Convert to NumPy array and Rescale (Volts to uV)
    signals = raw.get_data()
    if np.max(np.abs(signals)) <= 1e-4:
        logging.debug("Rescaling signals from Volts to microVolts...")
        signals = signals * 1e6

    # Ensure raw data length matches hypnogram
    samples_per_epoch = int(fs * 30.0)
    expected_samples = total_epochs * samples_per_epoch

    if signals.shape[1] < expected_samples:
        pad_width = expected_samples - signals.shape[1]
        logging.warning(
            f"Data length mismatch for {subject_id}: Signal length ({signals.shape[1]}) < expected ({expected_samples}). Zero-padding {pad_width} samples at the end."
        )
        signals = np.pad(
            signals, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
        )

    # Truncate strictly to the epochs defined by the hypnogram
    signals = signals[:, :expected_samples]

    # Safely identify EEG and EOG channel indices
    ch_names_picked = raw.ch_names
    eeg_idx = ch_names_picked.index(selected_channels[0])
    eog_idx = ch_names_picked.index(selected_channels[1])

    # Reshape to (Epochs, Samples, Channels) -> (N, 3000, 2)
    eeg_data = signals[eeg_idx].reshape(total_epochs, samples_per_epoch)
    eog_data = signals[eog_idx].reshape(total_epochs, samples_per_epoch)
    data = np.stack((eeg_data, eog_data), axis=2)  # (N, 3000, 2)

    # Build label and y
    label_map = {"W": 1, "N1": 2, "N2": 3, "N3": 4, "R": 5, "NS": 0}
    numeric_labels = hypno_df["Normalized_Stage"].map(label_map).values

    # Find valid indices (exclude NS / 0)
    valid_idx = numeric_labels != 0

    if not np.any(valid_idx):
        logging.warning(
            f"Skipping {subject_id}: No valid sleep stages left after removing NS."
        )
        return False

    # Filter data and labels
    data = data[valid_idx]
    label = numeric_labels[valid_idx].reshape(-1, 1).astype(np.int8)  # (N, 1)

    # Create one-hot encoding for y (N, 5)
    n_epochs_valid = len(label)
    y = np.zeros((n_epochs_valid, 5), dtype=np.int8)
    y[np.arange(n_epochs_valid), (label.flatten() - 1)] = 1

    # Output save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, f"{subject_id}_processed.mat")

    savemat(
        out_file,
        {"data": data.astype(np.float32), "label": label, "y": y},
        do_compression=True,
    )
    logging.debug(
        f"Successfully processed {subject_id}. Saved to {out_file} (data: {data.shape}, label: {label.shape}, y: {y.shape})"
    )

    return True


def main():
    logging.info("=== Starting EEG Data Preprocessing Pipeline ===")
    logging.info(f"Target Base Directory: {BASE_DIR}")

    if not os.path.exists(BASE_DIR):
        logging.error(f"Base Directory '{BASE_DIR}' does not exist.")
        return

    # Pipeline iterates through 100 folders (all folders) within BASE_DIR
    subjects = [
        d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))
    ]
    logging.info(f"Found {len(subjects)} subjects in base directory.")

    processed_count = 0
    skipped_count = 0

    for subject_id in subjects:
        subject_dir = os.path.join(BASE_DIR, subject_id)

        try:
            success = process_subject(subject_id, subject_dir)
            if success:
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            logging.error(
                f"Unexpected error while processing subject {subject_id}: {e}",
                exc_info=True,
            )
            skipped_count += 1

        # Memory Efficiency requirement
        gc.collect()

    logging.info("=== Pipeline Execution Summary ===")
    logging.info(f"Processed: {processed_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
