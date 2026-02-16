#!/usr/bin/env python
# coding: utf-8

"""
Custom EDF/CSV Preprocessing Script for mulEEG
Based on SHHS preprocessing logic

Converts EDF signals and CSV hypnogram data into mulEEG-compatible .npz format
"""

import numpy as np
import os
import pandas as pd
from mne.io import read_raw_edf


def parse_hypnogram_csv(csv_path, epoch_sec_size=30):
    """
    Parse hypnogram CSV file and convert sleep stages to numeric labels
    
    Args:
        csv_path: Path to CSV file containing sleep stage labels
        epoch_sec_size: Duration of each epoch in seconds (default: 30)
    
    Returns:
        labels: numpy array of numeric sleep stage labels (0-4)
    
    Label mapping:
        0: Wake (W)
        1: N1
        2: N2
        3: N3/N4
        4: REM
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Try to find the sleep stage column
    # Common column names: 'stage', 'Sleep Stage', 'label', 'hypnogram', 'annotation'
    stage_col = None
    possible_cols = ['stage', 'Sleep Stage', 'label', 'hypnogram', 'annotation', 'event']
    
    for col in possible_cols:
        if col in df.columns:
            stage_col = col
            break
    
    if stage_col is None:
        # If no standard column found, use the first column
        stage_col = df.columns[0]
        print(f"Warning: Using first column '{stage_col}' as sleep stage column")
    
    print(f"Using column '{stage_col}' for sleep stages")
    
    # Extract sleep stages
    stages = df[stage_col].values
    
    # Convert to numeric labels
    labels = []
    for stage in stages:
        stage_str = str(stage).strip().upper()
        
        # Map to numeric labels
        # Wake (0)
        if stage_str in ['W', 'WK', 'WAKE', '0']:
            labels.append(0)
        # N1 (1)
        elif stage_str in ['N1', '1', 'S1']:
            labels.append(1)
        # N2 (2)
        elif stage_str in ['N2', '2', 'S2']:
            labels.append(2)
        # N3/N4 (3)
        elif stage_str in ['N3', 'N4', '3', '4', 'S3', 'S4']:
            labels.append(3)  # Combine N3 and N4
        # REM (4)
        elif stage_str in ['REM', 'R', '5']:
            labels.append(4)
        # Additional numeric codes (some systems use 6=Movement, 7=Unknown, 8=Artifact)
        # Treat these as Wake (0) since they're not sleep stages
        elif stage_str in ['6', '7', '8']:
            print(f"Info: Sleep stage '{stage}' (movement/unknown/artifact) mapped to Wake (0)")
            labels.append(0)
        else:
            print(f"Warning: Unknown sleep stage '{stage}', treating as Wake (0)")
            labels.append(0)
    
    return np.array(labels, dtype=np.int32)


def preprocess_edf_csv(edf_path, hypnogram_path, output_path, 
                       select_channel=None, trim_wake_edges=True, 
                       edge_minutes=30, epoch_sec_size=30):
    """
    Preprocess EDF and CSV files into mulEEG-compatible format
    
    Args:
        edf_path: Path to EDF file
        hypnogram_path: Path to CSV hypnogram file
        output_path: Path to save output .npz file
        select_channel: Specific EEG channel to extract (None = auto-select first EEG)
        trim_wake_edges: Whether to trim wake periods at edges
        edge_minutes: Minutes to extend before/after sleep period
        epoch_sec_size: Epoch duration in seconds
    """
    print(f"\nProcessing: {edf_path}")
    
    # Read EDF file WITHOUT preloading all data first (memory optimization)
    raw = read_raw_edf(edf_path, preload=False, stim_channel=None, verbose=False)
    sampling_rate = raw.info['sfreq']
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Available channels: {raw.ch_names}")
    print(f"Total channels: {len(raw.ch_names)}")
    
    # Select EEG channel
    if select_channel is None:
        # Auto-select first EEG channel
        eeg_channels = [ch for ch in raw.info['ch_names'] if 'EEG' in ch.upper()]
        if len(eeg_channels) == 0:
            # If no EEG channel found, use first channel
            select_channel = raw.info['ch_names'][0]
            print(f"Warning: No EEG channel found, using first channel: {select_channel}")
        else:
            select_channel = eeg_channels[0]
            print(f"Auto-selected EEG channel: {select_channel}")
    else:
        print(f"Using specified channel: {select_channel}")
    
    # ⭐ MEMORY OPTIMIZATION: Pick only the channel we need BEFORE loading data
    # This prevents loading all channels into memory
    print(f"Selecting channel: {select_channel}")
    raw.pick_channels([select_channel])
    
    # ⭐ MEMORY OPTIMIZATION: Now load only the selected channel
    print("Loading channel data...")
    raw.load_data()
    
    # Get original sampling rate
    original_sfreq = raw.info['sfreq']
    print(f"Original sampling rate: {original_sfreq} Hz")
    target_sfreq = 100.0  # mulEEG model expects 100Hz (3000 points per 30s epoch)
    
    # ⭐ RESAMPLING: Resample to 100Hz if necessary for model compatibility
    if original_sfreq != target_sfreq:
        print(f"Resampling from {original_sfreq} Hz to {target_sfreq} Hz...")
        raw.resample(target_sfreq, npad='auto', verbose=False)
        print(f"Resampling completed. New sampling rate: {raw.info['sfreq']} Hz")
    else:
        print(f"Sampling rate is already {target_sfreq} Hz, no resampling needed")
    
    # Update sampling_rate variable after resampling
    sampling_rate = raw.info['sfreq']
    
    # ⭐ MEMORY OPTIMIZATION: Use get_data() instead of to_data_frame()
    # This avoids creating a pandas DataFrame which uses extra memory
    raw_ch = raw.get_data()[0]  # Get first (and only) channel
    print(f"Signal shape: {raw_ch.shape}")
    
    # Parse hypnogram
    labels = parse_hypnogram_csv(hypnogram_path, epoch_sec_size)
    print(f"Number of labels: {len(labels)}")
    
    # Verify that we can split into epochs
    samples_per_epoch = int(epoch_sec_size * sampling_rate)
    expected_samples = len(labels) * samples_per_epoch
    
    if len(raw_ch) != expected_samples:
        print(f"Warning: Signal length ({len(raw_ch)}) doesn't match expected ({expected_samples})")
        print(f"         Based on {len(labels)} labels × {samples_per_epoch} samples/epoch")
        
        # Trim or pad to match
        if len(raw_ch) > expected_samples:
            print(f"         Trimming signal to {expected_samples} samples")
            raw_ch = raw_ch[:expected_samples]
        else:
            print(f"         Padding signal to {expected_samples} samples")
            raw_ch = np.pad(raw_ch, (0, expected_samples - len(raw_ch)), mode='constant')
    
    # Split into epochs
    n_epochs = len(labels)
    x = raw_ch.reshape(n_epochs, samples_per_epoch).astype(np.float32)
    y = labels.astype(np.int32)
    
    print(f"Epochs shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: W={np.sum(y==0)}, N1={np.sum(y==1)}, N2={np.sum(y==2)}, N3={np.sum(y==3)}, REM={np.sum(y==4)}")
    
    # Trim wake edges (similar to SHHS preprocessing)
    if trim_wake_edges:
        nw_idx = np.where(y != 0)[0]
        if len(nw_idx) > 0:
            start_idx = nw_idx[0] - (edge_minutes * 2)  # 2 epochs per minute (30s each)
            end_idx = nw_idx[-1] + (edge_minutes * 2)
            
            if start_idx < 0:
                start_idx = 0
            if end_idx >= len(y):
                end_idx = len(y) - 1
            
            select_idx = np.arange(start_idx, end_idx + 1)
            print(f"Data before trimming: {x.shape}, {y.shape}")
            x = x[select_idx]
            y = y[select_idx]
            print(f"Data after trimming: {x.shape}, {y.shape}")
    
    # Save as .npz file
    save_dict = {
        "x": x,
        "y": y,
        "fs": sampling_rate
    }
    np.savez(output_path, **save_dict)
    print(f"Saved to: {output_path}")
    print("=" * 50)


def main():
    # ==================== CONFIGURATION ====================
    # แก้ไขค่าตัวแปรเหล่านี้ตามไฟล์ข้อมูลของคุณ
    
    # Path to input directory containing patient folders
    input_base_dir = "/home/nummm/Documents/CEPP/rawEEG"  # โฟลเดอร์หลักที่มี folder ของคนไข้
    
    # Output settings
    output_base_dir = "./output"                           # โฟลเดอร์หลักสำหรับบันทึกไฟล์ที่ประมวลผลแล้ว
                                                           # จะสร้าง subfolder แยกตาม patient ID
    
    # Processing limits
    max_patients = None                                      # จำนวนคนไข้สูงสุดที่จะประมวลผล (None = ทั้งหมด)
    
    # Channel selection
    select_channel = None                                  # ช่อง EEG ที่ต้องการ (None = เลือกอัตโนมัติ)
                                                           # ตัวอย่าง: "EEG C4-A1", "EEG Fpz-Cz"
    
    # Processing options
    trim_wake_edges = True                                 # ตัดช่วง wake ที่ขอบออก (True/False)
    edge_minutes = 30                                      # จำนวนนาทีที่ขยายก่อน/หลังช่วงนอน
    epoch_sec_size = 30                                    # ขนาด epoch (วินาที)
    
    # =======================================================
    
    print("=" * 70)
    print("EDF/CSV Batch Preprocessing for mulEEG")
    print("=" * 70)
    print(f"Input directory: {input_base_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Max patients: {max_patients if max_patients else 'All'}")
    print("=" * 70)
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get list of patient folders
    if not os.path.exists(input_base_dir):
        print(f"Error: Input directory does not exist: {input_base_dir}")
        return
    
    patient_folders = [f for f in os.listdir(input_base_dir) 
                      if os.path.isdir(os.path.join(input_base_dir, f))]
    patient_folders.sort()
    
    print(f"\nFound {len(patient_folders)} patient folders")
    
    # Track processing
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process each patient folder
    for patient_id in patient_folders:
        # Check if we've reached the limit
        if max_patients is not None and processed_count >= max_patients:
            print(f"\n✓ Reached maximum patient limit ({max_patients}). Stopping.")
            break
        
        patient_input_dir = os.path.join(input_base_dir, patient_id)
        patient_output_dir = os.path.join(output_base_dir, patient_id)
        
        # Check if already processed (skip if output folder exists)
        if os.path.exists(patient_output_dir):
            print(f"\n⊘ Skipping {patient_id} - already processed (output folder exists)")
            skipped_count += 1
            continue
        
        # Expected file paths
        edf_file = os.path.join(patient_input_dir, "edf_signals.edf")
        hypnogram_file = os.path.join(patient_input_dir, "csv_hypnogram.csv")
        
        # Check if required files exist
        if not os.path.exists(edf_file):
            print(f"\n✗ Skipping {patient_id} - edf_signals.edf not found")
            error_count += 1
            continue
        
        if not os.path.exists(hypnogram_file):
            print(f"\n✗ Skipping {patient_id} - csv_hypnogram.csv not found")
            error_count += 1
            continue
        
        # Create patient output directory
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Output file path
        output_file = os.path.join(patient_output_dir, f"{patient_id}.npz")
        
        print(f"\n{'='*70}")
        print(f"Processing patient {processed_count + 1}: {patient_id}")
        print(f"{'='*70}")
        
        try:
            # Preprocess
            preprocess_edf_csv(
                edf_path=edf_file,
                hypnogram_path=hypnogram_file,
                output_path=output_file,
                select_channel=select_channel,
                trim_wake_edges=trim_wake_edges,
                edge_minutes=edge_minutes,
                epoch_sec_size=epoch_sec_size
            )
            processed_count += 1
            print(f"✓ Successfully processed {patient_id}")
            
        except Exception as e:
            print(f"\n✗ Error processing {patient_id}: {str(e)}")
            error_count += 1
            # Remove output directory if processing failed
            if os.path.exists(patient_output_dir):
                import shutil
                shutil.rmtree(patient_output_dir)
            continue
    
    # Print summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total patients found:    {len(patient_folders)}")
    print(f"Successfully processed:  {processed_count}")
    print(f"Skipped (already done):  {skipped_count}")
    print(f"Errors:                  {error_count}")
    print("=" * 70)



if __name__ == "__main__":
    seed = 123
    np.random.seed(seed)
    main()
