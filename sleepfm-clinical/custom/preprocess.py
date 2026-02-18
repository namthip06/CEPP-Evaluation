#!/usr/bin/env python3
"""
Preprocessing script to convert EDF files to HDF5 format for raw EEG data.

This script processes folders in /home/nummm/Documents/CEPP/rawEEG/
Each folder should contain:
- edf_signals.edf
- csv_hypnogram.csv
- csv_events.csv

The script will:
1. Convert EDF to HDF5 format
2. Process hypnogram CSV to create sleep stage labels
3. Save HDF5 file in the same folder
4. Skip folders without edf_signals.edf
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Add parent directory to path to import preprocessing module
sys.path.append("..")
sys.path.append("../sleepfm")
from preprocessing.preprocessing import EDFToHDF5Converter


def parse_time(time_str):
    """Parse time string in format 'HH:MM:SS AM/PM' to datetime object."""
    try:
        return datetime.strptime(time_str.strip(), "%I:%M:%S %p")
    except ValueError:
        # Try alternative formats
        try:
            return datetime.strptime(time_str.strip(), "%H:%M:%S")
        except ValueError:
            return None


def convert_hypnogram_to_labels(hypnogram_csv_path, output_csv_path, sampling_freq=128, epoch_duration=30):
    """
    Convert hypnogram CSV to sleep stage labels format expected by SleepFM.
    
    Args:
        hypnogram_csv_path: Path to csv_hypnogram.csv
        output_csv_path: Path to save the converted labels
        sampling_freq: Sampling frequency (default: 128 Hz)
        epoch_duration: Duration of each epoch in seconds (default: 30s)
    
    Expected input format:
        Epoch Number,Start Time,Sleep Stage
        1,5:51:06 AM,NS
        2,5:51:36 AM,NS
        
    Output format:
        Start,Stop,StageName,StageNumber
        0.0,30.0,Wake,0
        30.0,60.0,Stage 1,1
    """
    # Read hypnogram
    df = pd.read_csv(hypnogram_csv_path)
    
    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()
    
    # Map sleep stages to numbers
    # NS = Non-Sleep/Wake = 0
    # S1 = Stage 1 = 1
    # S2 = Stage 2 = 2
    # S3/S4 = Stage 3 = 3
    # REM = REM = 4
    stage_mapping = {
        'NS': (0, 'Wake'),
        'W': (0, 'Wake'),
        'Wake': (0, 'Wake'),
        'S1': (1, 'Stage 1'),
        'N1': (1, 'Stage 1'),
        'Stage 1': (1, 'Stage 1'),
        'S2': (2, 'Stage 2'),
        'N2': (2, 'Stage 2'),
        'Stage 2': (2, 'Stage 2'),
        'S3': (3, 'Stage 3'),
        'S4': (3, 'Stage 3'),
        'N3': (3, 'Stage 3'),
        'Stage 3': (3, 'Stage 3'),
        'REM': (4, 'REM'),
        'R': (4, 'REM'),
    }
    
    # Create output dataframe
    output_data = []
    
    for idx, row in df.iterrows():
        epoch_num = row['Epoch Number']
        sleep_stage = row['Sleep Stage'].strip()
        
        # Calculate start and stop times based on epoch number
        start_time = (epoch_num - 1) * epoch_duration
        stop_time = epoch_num * epoch_duration
        
        # Map sleep stage
        if sleep_stage in stage_mapping:
            stage_num, stage_name = stage_mapping[sleep_stage]
        else:
            # Default to Wake if unknown
            print(f"Warning: Unknown sleep stage '{sleep_stage}' at epoch {epoch_num}, defaulting to Wake")
            stage_num, stage_name = 0, 'Wake'
        
        output_data.append({
            'Start': start_time,
            'Stop': stop_time,
            'StageName': stage_name,
            'StageNumber': stage_num
        })
    
    # Create output dataframe
    output_df = pd.DataFrame(output_data)
    
    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"  ✓ Converted hypnogram: {len(output_df)} epochs")
    
    return output_df


def process_single_folder(folder_path, converter):
    """
    Process a single folder containing EDF and CSV files.
    
    Args:
        folder_path: Path to the folder containing the files
        converter: EDFToHDF5Converter instance
    
    Returns:
        True if successful, False otherwise
    """
    folder_name = os.path.basename(folder_path)
    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")
    
    # Check if edf_signals.edf exists
    edf_path = os.path.join(folder_path, "edf_signals.edf")
    if not os.path.exists(edf_path):
        print(f"  ⚠ Skipping: edf_signals.edf not found")
        return False
    
    # Check if hypnogram exists
    hypnogram_path = os.path.join(folder_path, "csv_hypnogram.csv")
    if not os.path.exists(hypnogram_path):
        print(f"  ⚠ Warning: csv_hypnogram.csv not found, continuing without labels")
        hypnogram_path = None
    
    # Define output paths
    hdf5_path = os.path.join(folder_path, f"{folder_name}.hdf5")
    labels_csv_path = os.path.join(folder_path, f"{folder_name}.csv")
    
    try:
        # Step 1: Convert EDF to HDF5
        print(f"  → Converting EDF to HDF5...")
        converter.convert(edf_path, hdf5_path)
        print(f"  ✓ HDF5 saved: {os.path.basename(hdf5_path)}")
        
        # Step 2: Convert hypnogram if available
        if hypnogram_path:
            print(f"  → Converting hypnogram to labels...")
            convert_hypnogram_to_labels(hypnogram_path, labels_csv_path)
            print(f"  ✓ Labels saved: {os.path.basename(labels_csv_path)}")
        
        print(f"  ✅ Successfully processed: {folder_name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {folder_name}:")
        print(f"     {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main function to process all folders in rawEEG directory."""
    
    # Configuration
    raw_eeg_dir = "/home/nummm/Documents/CEPP/rawEEG"
    resample_rate = 128  # Hz
    
    print("\n" + "="*60)
    print("EDF to HDF5 Preprocessing Script")
    print("="*60)
    print(f"Source directory: {raw_eeg_dir}")
    print(f"Resample rate: {resample_rate} Hz")
    print("="*60)
    
    # Initialize converter
    converter = EDFToHDF5Converter(
        root_dir="/dummy_root",  # Not used for single file conversion
        target_dir="/dummy_target",  # Not used for single file conversion
        resample_rate=resample_rate
    )
    
    # Get all folders in rawEEG directory
    if not os.path.exists(raw_eeg_dir):
        print(f"Error: Directory not found: {raw_eeg_dir}")
        return
    
    folders = sorted([
        os.path.join(raw_eeg_dir, d) 
        for d in os.listdir(raw_eeg_dir) 
        if os.path.isdir(os.path.join(raw_eeg_dir, d))
    ])
    
    print(f"\nFound {len(folders)} folders to process")
    
    # Process each folder one at a time
    success_count:int = 0
    skip_count:int = 0
    error_count:int = 0
    
    for i, folder_path in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}]", end=" ")
        
        result = process_single_folder(folder_path, converter)
        
        if result:
            success_count += 1
        elif result is False and "Skipping" in str(result):
            skip_count += 1
        else:
            error_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total folders: {len(folders)}")
    print(f"✅ Successfully processed: {success_count}")
    print(f"⚠  Skipped (no EDF): {skip_count}")
    print(f"❌ Errors: {error_count}")
    print("="*60)
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
