#!/usr/bin/env python3
"""
Preprocessing script for rawEEG data
Processes data from /home/nummm/Documents/CEPP/rawEEG/[id]/ folders
Outputs to custom/preprocessing_output

Folder structure:
    rawEEG/
    ├── [id]/
    │   ├── csv_events.csv
    │   ├── csv_hypnogram.csv
    │   └── edf_signals.edf

Output: preprocessed FIF files with annotations
"""

import os
import sys
import mne
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Suppress MNE logging
mne.set_log_level('WARNING')

# Define absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_EEG_DIR = '/home/nummm/Documents/CEPP/rawEEG'
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'preprocessing_output'))


def parse_hypnogram(csv_path):
    """
    Parse hypnogram CSV file and create MNE annotations
    
    Parameters:
    -----------
    csv_path : str
        Path to csv_hypnogram.csv
    
    Returns:
    --------
    annotations : mne.Annotations
        Annotations object with sleep stages
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # Expected columns: 'Epoch Number', 'Start Time', 'Sleep Stage'
        # Handle variations in column names
        epoch_col = None
        time_col = None
        stage_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'epoch' in col_lower:
                epoch_col = col
            elif 'time' in col_lower:
                time_col = col
            elif 'stage' in col_lower:
                stage_col = col
        
        if not all([epoch_col, time_col, stage_col]):
            print(f"  Warning: Could not find required columns in {csv_path}")
            print(f"  Available columns: {df.columns.tolist()}")
            return None
        
        # Parse data
        onsets = []
        durations = []
        descriptions = []
        
        # Assume 30-second epochs (standard for sleep staging)
        epoch_duration = 30.0
        
        for idx, row in df.iterrows():
            onset = idx * epoch_duration  # Start time in seconds from beginning
            duration = epoch_duration
            
            # Map sleep stage to standard format
            stage = str(row[stage_col]).strip().upper()
            
            # Standardize sleep stage names
            stage_mapping = {
                'WK': 'Sleep stage W',
                'W': 'Sleep stage W',
                'WAKE': 'Sleep stage W',
                'N1': 'Sleep stage 1',
                '1': 'Sleep stage 1',
                'N2': 'Sleep stage 2',
                '2': 'Sleep stage 2',
                'N3': 'Sleep stage 3',
                '3': 'Sleep stage 3',
                'N4': 'Sleep stage 4',
                '4': 'Sleep stage 4',
                'REM': 'Sleep stage R',
                'R': 'Sleep stage R',
            }
            
            description = stage_mapping.get(stage, f'Sleep stage {stage}')
            
            onsets.append(onset)
            durations.append(duration)
            descriptions.append(description)
        
        # Create annotations
        annotations = mne.Annotations(onset=onsets, 
                                      duration=durations, 
                                      description=descriptions)
        
        print(f"  Parsed {len(annotations)} sleep stage annotations")
        return annotations
        
    except Exception as e:
        print(f"  Error parsing hypnogram: {e}")
        return None


def preprocess_single_subject(subject_id, raw_dir, output_dir):
    """
    Preprocess a single subject's EEG data
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (folder name)
    raw_dir : str
        Path to rawEEG directory
    output_dir : str
        Path to output directory
    
    Returns:
    --------
    success : bool
        True if processing was successful
    """
    subject_path = os.path.join(raw_dir, subject_id)
    
    # Define file paths
    edf_path = os.path.join(subject_path, 'edf_signals.edf')
    hypnogram_path = os.path.join(subject_path, 'csv_hypnogram.csv')
    events_path = os.path.join(subject_path, 'csv_events.csv')
    
    # Check if EDF file exists
    if not os.path.exists(edf_path):
        print(f"  Skipping {subject_id}: edf_signals.edf not found")
        return False
    
    print(f"\nProcessing subject: {subject_id}")
    print(f"  EDF file: {edf_path}")
    
    try:
        # 1. Read EDF file
        print("  Loading EDF file...")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        print(f"  Original sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} seconds ({raw.times[-1]/60:.1f} minutes)")
        print(f"  Channels: {len(raw.ch_names)}")
        
        # 2. Pick only EEG channels
        try:
            raw.pick_types(eeg=True, exclude='bads')
            print(f"  Selected {len(raw.ch_names)} EEG channels")
        except Exception as e:
            print(f"  Warning: Could not pick EEG channels automatically: {e}")
            print(f"  Using all channels")
        
        # 3. Apply filters (band-pass 0.5-40 Hz, notch 50/60 Hz)
        print("  Applying filters...")
        try:
            raw.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)
            raw.notch_filter(freqs=[50, 60], verbose=False)
        except Exception as e:
            print(f"  Warning: Filtering failed: {e}")
        
        # 4. Resample to 256 Hz (BENDR standard)
        target_sfreq = 256
        if raw.info['sfreq'] != target_sfreq:
            print(f"  Resampling to {target_sfreq} Hz...")
            raw.resample(target_sfreq, verbose=False)
        
        # 5. Check and fix bad channels
        print("  Checking for bad channels...")
        try:
            data = raw.get_data()
            channel_stds = np.std(data, axis=1)
            median_std = np.median(channel_stds)
            bad_threshold = 5  # 5x median
            
            bad_channels = []
            for i, std in enumerate(channel_stds):
                if std > bad_threshold * median_std or std < median_std / bad_threshold:
                    bad_channels.append(raw.ch_names[i])
            
            if bad_channels:
                print(f"  Found bad channels: {bad_channels}")
                raw.info['bads'] = bad_channels
                raw.interpolate_bads(reset_bads=True, verbose=False)
            else:
                print("  No bad channels detected")
        except Exception as e:
            print(f"  Warning: Bad channel detection failed: {e}")
        
        # 6. Parse and add hypnogram annotations
        if os.path.exists(hypnogram_path):
            print("  Parsing hypnogram...")
            annotations = parse_hypnogram(hypnogram_path)
            if annotations is not None:
                raw.set_annotations(annotations)
                print("  Added sleep stage annotations")
        else:
            print(f"  Warning: Hypnogram file not found: {hypnogram_path}")
        
        # 7. Save preprocessed data
        output_file = os.path.join(output_dir, f'{subject_id}_preprocessed.fif')
        print(f"  Saving to: {output_file}")
        raw.save(output_file, overwrite=True, verbose=False)
        
        # Clean up to free memory
        del raw
        del data
        
        print(f"  ✓ Successfully processed {subject_id}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to process all subjects in rawEEG directory
    """
    print("="*70)
    print("RawEEG Preprocessing Script")
    print("="*70)
    print(f"Input directory: {RAW_EEG_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory created: {OUTPUT_DIR}")
    
    # Check if rawEEG directory exists
    if not os.path.exists(RAW_EEG_DIR):
        print(f"\nError: rawEEG directory not found: {RAW_EEG_DIR}")
        sys.exit(1)
    
    # Get list of subject IDs (folder names)
    try:
        all_items = os.listdir(RAW_EEG_DIR)
        subject_ids = [item for item in all_items 
                      if os.path.isdir(os.path.join(RAW_EEG_DIR, item))]
        subject_ids.sort()  # Sort for consistent processing order
    except Exception as e:
        print(f"\nError reading rawEEG directory: {e}")
        sys.exit(1)
    
    print(f"\nFound {len(subject_ids)} subject folders")
    
    if len(subject_ids) == 0:
        print("No subject folders found!")
        sys.exit(1)
    
    # Process each subject sequentially (no batch/parallel processing)
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, subject_id in enumerate(subject_ids, 1):
        print(f"\n{'='*70}")
        print(f"Progress: {i}/{len(subject_ids)}")
        print(f"{'='*70}")
        
        result = preprocess_single_subject(subject_id, RAW_EEG_DIR, OUTPUT_DIR)
        
        if result:
            success_count += 1
        elif result is False:
            # Check if it was skipped or error
            edf_path = os.path.join(RAW_EEG_DIR, subject_id, 'edf_signals.edf')
            if not os.path.exists(edf_path):
                skip_count += 1
            else:
                error_count += 1
    
    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total subjects: {len(subject_ids)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (no EDF): {skip_count}")
    print(f"Errors: {error_count}")
    print("="*70)
    print(f"\nPreprocessed files saved to: {OUTPUT_DIR}")
    
    # List output files
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.fif')]
    print(f"Total output files: {len(output_files)}")


if __name__ == '__main__':
    main()
