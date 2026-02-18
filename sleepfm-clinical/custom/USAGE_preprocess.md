# Preprocessing Guide for Raw EEG Data

## Purpose

This guide explains how to use the `preprocess.py` script. The script is designed to automate the preprocessing of raw PSG data for use with the SleepFM pipeline.

Its primary objectives are:
1.  **Format Conversion**: Convert raw PSG recordings from European Data Format (EDF) to HDF5 format, which is optimized for fast data loading during deep learning training.
2.  **Label Standardization**: Parse sleep stage annotations (hypnograms) from `csv_hypnogram.csv` and convert them into a standardized CSV format (`Start`, `Stop`, `StageName`, `StageNumber`) required by SleepFM.
3.  **Data Organization**: Ensure outputs are saved correctly alongside the input files, maintaining the folder structure.

## Folder Structure

Your raw EEG data should be organized as follows:

```
/home/nummm/Documents/CEPP/rawEEG/
├── 00000358-159547/
│   ├── edf_signals.edf
│   ├── csv_hypnogram.csv
│   └── csv_events.csv
├── 00000359-159547/
│   ├── edf_signals.edf
│   ├── csv_hypnogram.csv
│   └── csv_events.csv
└── ... (more folders)
```

## Required Files

Each folder should contain:
- **edf_signals.edf** (required): The raw PSG recording in EDF format
- **csv_hypnogram.csv** (optional): Sleep stage annotations
- **csv_events.csv** (optional): Sleep events

## Hypnogram Format

The `csv_hypnogram.csv` file should have the following format:

```csv
Epoch Number,Start Time,Sleep Stage
1,5:51:06 AM,NS
2,5:51:36 AM,NS
3,5:52:06 AM,S1
4,5:52:36 AM,S2
```

### Sleep Stage Mapping

The script automatically maps sleep stages to numbers:

| Input Stage | Stage Number | Stage Name |
|-------------|--------------|------------|
| NS, W, Wake | 0 | Wake |
| S1, N1, Stage 1 | 1 | Stage 1 |
| S2, N2, Stage 2 | 2 | Stage 2 |
| S3, S4, N3, Stage 3 | 3 | Stage 3 |
| REM, R | 4 | REM |

## Running the Preprocessing Script

### Prerequisites

1. Activate the SleepFM environment:
   ```bash
   conda activate sleepfm
   ```

2. Navigate to the custom directory:
   ```bash
   cd /home/nummm/Documents/CEPP/sleepfm-clinical/custom
   ```

### Execute the Script

```bash
python preprocess.py
```

The script will:
1. Find all folders in `/home/nummm/Documents/CEPP/rawEEG/`
2. Process each folder **one at a time** (no batch/parallel processing)
3. Skip folders without `edf_signals.edf`
4. Convert EDF files to HDF5 format
5. Convert hypnogram CSV to sleep stage labels
6. Save outputs in the same folder

### Output Files

For each processed folder (e.g., `00000358-159547/`), the script creates:

- **[id].hdf5**: Preprocessed PSG data in HDF5 format
  - Example: `00000358-159547.hdf5`
- **[id].csv**: Sleep stage labels in SleepFM format
  - Example: `00000358-159547.csv`

### Output Label Format

The generated CSV file will have the following format:

```csv
Start,Stop,StageName,StageNumber
0.0,30.0,Wake,0
30.0,60.0,Wake,0
60.0,90.0,Stage 1,1
90.0,120.0,Stage 2,2
```

## Configuration

You can modify the following parameters in `preprocess.py`:

- **resample_rate**: Target sampling frequency (default: 128 Hz)
- **epoch_duration**: Duration of each sleep epoch (default: 30 seconds)

## Troubleshooting

### Missing EDF File
If a folder doesn't have `edf_signals.edf`, the script will skip it and continue with the next folder.

### Missing Hypnogram
If `csv_hypnogram.csv` is missing, the script will still convert the EDF file but won't create sleep stage labels.

### Unknown Sleep Stages
If the hypnogram contains unknown sleep stage codes, the script will default them to "Wake" (0) and print a warning.

### Memory Issues
The script processes folders **one at a time** to avoid memory issues. Each folder is fully processed before moving to the next one.

## Next Steps

After preprocessing, you can use the generated HDF5 files with the SleepFM pipeline:

1. **Generate embeddings**: Use `demo.ipynb` Part 1
2. **Sleep staging**: Use `demo.ipynb` Part 2
3. **Disease prediction**: Use `demo.ipynb` Part 3

See [README.md](README.md) for more details on the SleepFM pipeline.

## Example Usage

```bash
# Navigate to custom directory
cd /home/nummm/Documents/CEPP/sleepfm-clinical/custom

# Run preprocessing
python preprocess.py
```

Expected output:
```
============================================================
EDF to HDF5 Preprocessing Script
============================================================
Source directory: /home/nummm/Documents/CEPP/rawEEG
Resample rate: 128 Hz
============================================================

Found 100 folders to process

[1/100] ============================================================
Processing: 00000358-159547
============================================================
  → Converting EDF to HDF5...
  ✓ HDF5 saved: 00000358-159547.hdf5
  → Converting hypnogram to labels...
  ✓ Converted hypnogram: 1234 epochs
  ✓ Labels saved: 00000358-159547.csv
  ✅ Successfully processed: 00000358-159547

...

============================================================
PROCESSING SUMMARY
============================================================
Total folders: 100
✅ Successfully processed: 95
⚠  Skipped (no EDF): 3
❌ Errors: 2
============================================================

Preprocessing complete!
```
