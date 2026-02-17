# Custom EEG Preprocessing Configuration

This directory contains scripts for preprocessing custom sleep staging EEG data.

## File Structure

```
custom/
├── preprocess_sleep_staging.py    # Main preprocessing script
├── preprocessing_README.md         # General preprocessing documentation
└── USAGE.md                        # This file
```

## Input Data Structure

Your raw EEG data should be organized as follows:

```
/home/nummm/Documents/CEPP/rawEEG/
├── [id_1]/
│   ├── edf_signals.edf
│   ├── csv_hypnogram.csv
│   └── csv_events.csv
├── [id_2]/
│   ├── edf_signals.edf
│   ├── csv_hypnogram.csv
│   └── csv_events.csv
...
└── [id_100]/
    ├── edf_signals.edf
    ├── csv_hypnogram.csv
    └── csv_events.csv
```

## Output Data Structure

Preprocessed data will be saved to:

```
custom/preprocessing_output/
├── seq/
│   ├── [id_1]/
│   │   ├── [id_1]_seq_0.npy
│   │   ├── [id_1]_seq_1.npy
│   │   └── ...
│   └── [id_2]/
│       └── ...
└── labels/
    ├── [id_1]/
    │   ├── [id_1]_label_0.npy
    │   ├── [id_1]_label_1.npy
    │   └── ...
    └── [id_2]/
        └── ...
```

## How to Use

### 1. Configure the Script

Open `preprocess_sleep_staging.py` and modify the configuration section if needed:

```python
# Preprocessing parameters
TARGET_SFREQ = 200          # Target sampling frequency (Hz)
EPOCH_LENGTH = 30           # Epoch length in seconds
SEQ_LENGTH = 20             # Number of epochs per sequence

# Filtering parameters
BANDPASS_LOW = 0.3          # Bandpass filter low cutoff (Hz)
BANDPASS_HIGH = 35          # Bandpass filter high cutoff (Hz)
NOTCH_FREQ = 50             # Notch filter frequency (50 or 60 Hz)

# Sleep stage mapping - IMPORTANT: Adjust based on your csv_hypnogram.csv
SLEEP_STAGE_MAP = {
    'WK': 0,   # Wake
    'W': 0,    # Wake (alternative)
    'N1': 1,   # NREM Stage 1
    'N2': 2,   # NREM Stage 2
    'N3': 3,   # NREM Stage 3
    'REM': 4,  # REM
    'R': 4,    # REM (alternative)
}

# Channel selection
CHANNELS_TO_USE = None  # None = use all channels
# Or specify channels:
# CHANNELS_TO_USE = ['C3-A2', 'C4-A1', 'F3-A2', 'F4-A1', 'O1-A2', 'O2-A1']
```

### 2. Check Your Hypnogram Format

Make sure your `csv_hypnogram.csv` format matches the expected format. Example:

```csv
Epoch Number,Start Time,Sleep Stage
1,9:18:51 PM,WK
2,9:19:21 PM,WK
3,9:19:51 PM,N1
4,9:20:21 PM,N2
...
```

The script will automatically detect the column containing sleep stages (looks for "stage" in column name).

### 3. Run the Script

```bash
cd /home/nummm/Documents/CEPP/EEGMamba/custom
python preprocess_sleep_staging.py
```

### 4. Monitor Progress

The script will:
- Process each patient ID sequentially (low memory usage)
- Skip patients without `edf_signals.edf`
- Display progress with tqdm progress bar
- Show detailed information for each patient
- Print summary at the end

Example output:
```
Processing patient ID: 001
  Loading EDF file...
  Original sampling frequency: 256.0 Hz
  Applying bandpass filter (0.3-35 Hz)...
  Applying notch filter (50 Hz)...
  Resampling to 200 Hz...
  Setting average reference...
  Extracting data...
  Data shape: (720000, 6) (samples, channels)
  Created 120 epochs of 30s each
  Created 6 sequences
  Saving preprocessed data...
  Successfully processed patient 001
```

## Important Notes

### Memory Usage
- The script processes **one patient at a time** to minimize memory usage
- No batch or parallel processing
- Suitable for systems with limited RAM

### Missing Files
- If `edf_signals.edf` is not found, the patient will be **skipped**
- If `csv_hypnogram.csv` is not found, the patient will be **skipped**
- The script will continue processing other patients

### Data Validation
- The script checks if the number of epochs matches the number of labels
- If there's a mismatch, it will trim to the minimum length
- Incomplete epochs (not fitting the sequence length) are removed

### Channel Selection
- By default, the script uses **all available channels** in the EDF file
- You can specify specific channels by modifying `CHANNELS_TO_USE`
- If specified channels are not found, the script will use all channels

### Sleep Stage Mapping
- **CRITICAL**: Make sure `SLEEP_STAGE_MAP` matches your hypnogram format
- Unknown sleep stages will be skipped with a warning
- Common formats: 'WK'/'W', 'N1', 'N2', 'N3', 'REM'/'R'

## Output Format

Each preprocessed sequence has the shape:
- **Sequences**: `(seq_length, n_channels, epoch_samples)` = `(20, n_channels, 6000)`
- **Labels**: `(seq_length,)` = `(20,)`

Where:
- `seq_length` = 20 epochs per sequence
- `n_channels` = number of EEG channels
- `epoch_samples` = 6000 samples (30 seconds × 200 Hz)

## Troubleshooting

### Error: "Could not find sleep stage column"
- Check your `csv_hypnogram.csv` format
- Make sure there's a column with "stage" in its name
- Modify the column detection logic in `load_hypnogram()` if needed

### Error: "Mismatch between epochs and labels"
- This is normal if your recording length doesn't match the hypnogram exactly
- The script will automatically trim to the minimum length

### Error: "None of the specified channels found"
- Check your EDF file's channel names
- Use `CHANNELS_TO_USE = None` to use all channels
- Or update `CHANNELS_TO_USE` with correct channel names

### Low number of sequences
- Check if your recordings are long enough
- Each sequence requires 20 epochs × 30 seconds = 10 minutes of data
- Incomplete sequences are removed

## Next Steps

After preprocessing, you can use the preprocessed data with the EEGMamba model:

1. Create a custom dataset loader (similar to `datasets/isruc_dataset.py`)
2. Point it to `custom/preprocessing_output/`
3. Use the model for training or evaluation

See the main `preprocessing_README.md` for more details on the preprocessing pipeline.
