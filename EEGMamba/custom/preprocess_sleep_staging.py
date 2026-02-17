"""
Custom EEG Preprocessing Script for Sleep Staging
Processes raw EEG data from /home/nummm/Documents/CEPP/rawEEG/[id]/ directory structure
Outputs preprocessed data to custom/preprocessing_output/

Input structure:
- /home/nummm/Documents/CEPP/rawEEG/[id]/
  - edf_signals.edf
  - csv_hypnogram.csv
  - csv_events.csv

Output structure:
- custom/preprocessing_output/seq/[id]/
- custom/preprocessing_output/labels/[id]/
"""

import os
import numpy as np
import mne
from mne.io import read_raw_edf
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# Base directory (absolute path)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
raw_eeg_dir = '/home/nummm/Documents/CEPP/rawEEG'
output_dir = os.path.join(base_dir, 'custom', 'preprocessing_output')

# Create output directories
seq_output_dir = os.path.join(output_dir, 'seq')
label_output_dir = os.path.join(output_dir, 'labels')
os.makedirs(seq_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)

# Preprocessing parameters (following sleep staging example)
TARGET_SFREQ = 200  # Target sampling frequency in Hz
EPOCH_LENGTH = 30  # Epoch length in seconds
EPOCH_SAMPLES = EPOCH_LENGTH * TARGET_SFREQ  # 6000 samples per epoch
SEQ_LENGTH = 20  # Number of epochs per sequence

# Filtering parameters
BANDPASS_LOW = 0.3  # Hz
BANDPASS_HIGH = 35  # Hz
NOTCH_FREQ = 50  # Hz (use 60 for US data)

# Sleep stage mapping (adjust based on your csv_hypnogram.csv format)
# Example mapping - modify according to your data
SLEEP_STAGE_MAP = {
    'WK': 0,  # Wake
    'W': 0,   # Wake (alternative)
    'N1': 1,  # NREM Stage 1
    'N2': 2,  # NREM Stage 2
    'N3': 3,  # NREM Stage 3
    'REM': 4, # REM
    'R': 4,   # REM (alternative)
}

# Channel selection (adjust based on your EDF file channels)
# Example: Common sleep staging channels
# You may need to modify this based on your actual channel names
CHANNELS_TO_USE = None  # None = use all channels, or specify list like:
# CHANNELS_TO_USE = ['C3-A2', 'C4-A1', 'F3-A2', 'F4-A1', 'O1-A2', 'O2-A1']


# ============================================================================
# Helper Functions
# ============================================================================

def load_hypnogram(hypnogram_path):
    """
    Load hypnogram from CSV file and convert to epoch labels
    
    Args:
        hypnogram_path: Path to csv_hypnogram.csv
        
    Returns:
        List of sleep stage labels (integers)
    """
    try:
        df = pd.read_csv(hypnogram_path)
        
        # Extract sleep stage column (adjust column name if needed)
        # Common column names: 'Sleep Stage', 'Stage', 'sleep_stage'
        stage_column = None
        for col in df.columns:
            if 'stage' in col.lower():
                stage_column = col
                break
        
        if stage_column is None:
            print(f"Warning: Could not find sleep stage column in {hypnogram_path}")
            return None
        
        # Convert sleep stages to integer labels
        labels = []
        for stage in df[stage_column]:
            stage_str = str(stage).strip()
            if stage_str in SLEEP_STAGE_MAP:
                labels.append(SLEEP_STAGE_MAP[stage_str])
            else:
                print(f"Warning: Unknown sleep stage '{stage_str}', skipping")
                labels.append(-1)  # Mark as unknown
        
        # Remove unknown labels
        labels = [l for l in labels if l != -1]
        
        return labels
    
    except Exception as e:
        print(f"Error loading hypnogram from {hypnogram_path}: {e}")
        return None


def preprocess_eeg(edf_path, channels=None):
    """
    Preprocess EEG data from EDF file
    
    Args:
        edf_path: Path to edf_signals.edf
        channels: List of channel names to use (None = use all)
        
    Returns:
        Preprocessed numpy array of shape (n_epochs, n_channels, epoch_samples)
    """
    try:
        # Load raw EEG data
        print(f"  Loading EDF file...")
        raw = read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Select channels if specified
        if channels is not None:
            available_channels = raw.ch_names
            channels_to_pick = [ch for ch in channels if ch in available_channels]
            if len(channels_to_pick) == 0:
                print(f"  Warning: None of the specified channels found. Using all channels.")
            else:
                raw.pick_channels(channels_to_pick, ordered=True)
                print(f"  Selected {len(channels_to_pick)} channels")
        
        # Get original sampling frequency
        original_sfreq = raw.info['sfreq']
        print(f"  Original sampling frequency: {original_sfreq} Hz")
        
        # Apply filtering
        print(f"  Applying bandpass filter ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz)...")
        raw.filter(BANDPASS_LOW, BANDPASS_HIGH, fir_design='firwin', verbose=False)
        
        print(f"  Applying notch filter ({NOTCH_FREQ} Hz)...")
        raw.notch_filter(NOTCH_FREQ, verbose=False)
        
        # Resample to target frequency if needed
        if original_sfreq != TARGET_SFREQ:
            print(f"  Resampling to {TARGET_SFREQ} Hz...")
            raw.resample(TARGET_SFREQ, verbose=False)
        
        # Set average reference
        print(f"  Setting average reference...")
        raw.set_eeg_reference(ref_channels='average', verbose=False)
        
        # Get data as numpy array
        print(f"  Extracting data...")
        psg_array = raw.get_data().T  # Shape: (n_samples, n_channels)
        n_samples, n_channels = psg_array.shape
        print(f"  Data shape: {psg_array.shape} (samples, channels)")
        
        # Remove incomplete epochs
        remainder = n_samples % EPOCH_SAMPLES
        if remainder > 0:
            psg_array = psg_array[:-remainder, :]
            print(f"  Removed {remainder} incomplete samples")
        
        # Reshape into epochs
        n_epochs = psg_array.shape[0] // EPOCH_SAMPLES
        psg_array = psg_array.reshape(n_epochs, EPOCH_SAMPLES, n_channels)
        print(f"  Created {n_epochs} epochs of {EPOCH_LENGTH}s each")
        
        # Transpose to (n_epochs, n_channels, epoch_samples)
        psg_array = psg_array.transpose(0, 2, 1)
        
        return psg_array
    
    except Exception as e:
        print(f"  Error preprocessing EEG: {e}")
        return None


def process_patient(patient_id, raw_eeg_dir, seq_output_dir, label_output_dir):
    """
    Process a single patient's EEG data
    
    Args:
        patient_id: Patient ID (folder name)
        raw_eeg_dir: Base directory containing raw EEG data
        seq_output_dir: Output directory for sequences
        label_output_dir: Output directory for labels
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing patient ID: {patient_id}")
    
    # Define paths
    patient_dir = os.path.join(raw_eeg_dir, str(patient_id))
    edf_path = os.path.join(patient_dir, 'edf_signals.edf')
    hypnogram_path = os.path.join(patient_dir, 'csv_hypnogram.csv')
    
    # Check if EDF file exists
    if not os.path.exists(edf_path):
        print(f"  Skipping: edf_signals.edf not found")
        return False
    
    # Check if hypnogram exists
    if not os.path.exists(hypnogram_path):
        print(f"  Warning: csv_hypnogram.csv not found, skipping")
        return False
    
    # Load hypnogram
    print(f"  Loading hypnogram...")
    labels = load_hypnogram(hypnogram_path)
    if labels is None or len(labels) == 0:
        print(f"  Error: Could not load labels")
        return False
    
    # Preprocess EEG
    psg_array = preprocess_eeg(edf_path, channels=CHANNELS_TO_USE)
    if psg_array is None:
        print(f"  Error: Could not preprocess EEG")
        return False
    
    n_epochs, n_channels, epoch_samples = psg_array.shape
    
    # Check if number of epochs matches number of labels
    if n_epochs != len(labels):
        print(f"  Warning: Mismatch between epochs ({n_epochs}) and labels ({len(labels)})")
        # Trim to minimum length
        min_length = min(n_epochs, len(labels))
        psg_array = psg_array[:min_length]
        labels = labels[:min_length]
        print(f"  Trimmed to {min_length} epochs")
    
    # Group into sequences
    remainder = len(psg_array) % SEQ_LENGTH
    if remainder > 0:
        psg_array = psg_array[:-remainder]
        labels = labels[:-remainder]
        print(f"  Removed {remainder} epochs to fit sequence length")
    
    n_sequences = len(psg_array) // SEQ_LENGTH
    
    # Reshape into sequences
    # Shape: (n_sequences, seq_length, n_channels, epoch_samples)
    epochs_seq = psg_array.reshape(n_sequences, SEQ_LENGTH, n_channels, epoch_samples)
    labels_array = np.array(labels).reshape(n_sequences, SEQ_LENGTH)
    
    print(f"  Created {n_sequences} sequences")
    print(f"  Sequence shape: {epochs_seq.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    
    # Create output directories for this patient
    patient_seq_dir = os.path.join(seq_output_dir, str(patient_id))
    patient_label_dir = os.path.join(label_output_dir, str(patient_id))
    os.makedirs(patient_seq_dir, exist_ok=True)
    os.makedirs(patient_label_dir, exist_ok=True)
    
    # Save sequences and labels
    print(f"  Saving preprocessed data...")
    for i in range(n_sequences):
        # Save sequence
        seq_filename = os.path.join(patient_seq_dir, f'{patient_id}_seq_{i}.npy')
        np.save(seq_filename, epochs_seq[i])
        
        # Save label
        label_filename = os.path.join(patient_label_dir, f'{patient_id}_label_{i}.npy')
        np.save(label_filename, labels_array[i])
    
    print(f"  Successfully processed patient {patient_id}")
    return True


# ============================================================================
# Main Processing
# ============================================================================

def main():
    """
    Main function to process all patients
    """
    print("=" * 80)
    print("EEG Preprocessing Script for Sleep Staging")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Raw EEG directory: {raw_eeg_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Target sampling rate: {TARGET_SFREQ} Hz")
    print(f"  Epoch length: {EPOCH_LENGTH} seconds")
    print(f"  Sequence length: {SEQ_LENGTH} epochs")
    print(f"  Bandpass filter: {BANDPASS_LOW}-{BANDPASS_HIGH} Hz")
    print(f"  Notch filter: {NOTCH_FREQ} Hz")
    print(f"  Channels: {'All' if CHANNELS_TO_USE is None else CHANNELS_TO_USE}")
    print("=" * 80)
    
    # Get list of patient IDs (assuming IDs are folder names)
    if not os.path.exists(raw_eeg_dir):
        print(f"\nError: Raw EEG directory not found: {raw_eeg_dir}")
        return
    
    # Get all subdirectories (patient IDs)
    patient_ids = []
    for item in os.listdir(raw_eeg_dir):
        item_path = os.path.join(raw_eeg_dir, item)
        if os.path.isdir(item_path):
            patient_ids.append(item)
    
    patient_ids.sort()  # Sort for consistent processing order
    
    print(f"\nFound {len(patient_ids)} patient directories")
    
    if len(patient_ids) == 0:
        print("No patient directories found. Exiting.")
        return
    
    # Process each patient sequentially (low memory usage)
    successful = 0
    failed = 0
    skipped = 0
    
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        try:
            result = process_patient(patient_id, raw_eeg_dir, seq_output_dir, label_output_dir)
            if result:
                successful += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\nUnexpected error processing patient {patient_id}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Total patients: {len(patient_ids)}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (missing files): {skipped}")
    print(f"Failed (errors): {failed}")
    print(f"\nOutput saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
