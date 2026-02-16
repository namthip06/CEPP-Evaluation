# Embedding Generation Guide

## Overview
This guide explains how to extract sleep representations (embeddings) from raw EEG data using the pretrained SleepFM model.

## Prerequisites

### 1. Preprocessed Data
Before generating embeddings, you must first preprocess your raw EEG data using `preprocess.py`:

```bash
cd /home/nummm/Documents/CEPP/sleepfm-clinical/custom
python preprocess.py
```

This will create HDF5 files and CSV labels for each folder in `/home/nummm/Documents/CEPP/rawEEG/`.

### 2. Folder Structure
Each folder in `/home/nummm/Documents/CEPP/rawEEG/` should contain:
```
00000358-159547/
├── edf_signals.edf          # Original EDF file
├── csv_hypnogram.csv         # Sleep stage annotations
├── csv_events.csv            # Events (optional)
├── 00000358-159547.hdf5      # Preprocessed HDF5 (created by preprocess.py)
└── 00000358-159547.csv       # Sleep stage labels (created by preprocess.py)
```

## Usage

### Generate Embeddings

```bash
cd /home/nummm/Documents/CEPP/sleepfm-clinical/custom
python generate_embeddings.py
```

### What It Does

1. **Loads the pretrained model**: Uses `model_base` checkpoint from `../sleepfm/checkpoints/model_base`
2. **Processes EEG channels only**: Extracts embeddings from BAS (Brain Activity Signals) modality
3. **Sequential processing**: Processes one folder at a time to avoid RAM issues
4. **Skips missing files**: Automatically skips folders without HDF5 files
5. **Saves embeddings**: Creates two types of embeddings for each subject:
   - `{subject_id}_embeddings.hdf5` - Per-epoch embeddings
   - `{subject_id}_embeddings_5min.hdf5` - 5-minute aggregated embeddings

### Output

Embeddings are saved to:
```
/home/nummm/Documents/CEPP/sleepfm-clinical/custom/embeddings/
```

Each subject will have two files:
- `{subject_id}_embeddings.hdf5` - Detailed per-epoch embeddings
- `{subject_id}_embeddings_5min.hdf5` - Aggregated 5-minute embeddings

### Configuration

The script uses these default settings:
- **Model**: `../sleepfm/checkpoints/model_base`
- **Input directory**: `/home/nummm/Documents/CEPP/rawEEG`
- **Output directory**: `/home/nummm/Documents/CEPP/sleepfm-clinical/custom/embeddings`
- **Modality**: BAS (EEG channels only)
- **Batch size**: 1 (sequential processing)
- **Workers**: 0 (no parallel processing)

To modify these settings, edit the `main()` function in `generate_embeddings.py`.

## Memory Optimization

The script is designed to minimize RAM usage:
- **Sequential processing**: Processes one folder at a time (no batching)
- **Single worker**: No parallel data loading
- **CUDA cache clearing**: Clears GPU memory after each folder
- **EEG only**: Processes only BAS modality instead of all 4 modalities

## Troubleshooting

### "HDF5 file not found"
- Make sure you ran `preprocess.py` first
- Check that the folder contains `{folder_name}.hdf5`

### Out of Memory Error
- The script already uses minimal RAM settings
- Try closing other applications
- Consider processing fewer folders at a time by modifying the script

### CUDA Error
- The script will automatically fall back to CPU if CUDA is not available
- For CPU-only processing, it will be slower but should work

## Reading Embeddings

To read the generated embeddings in Python:

```python
import h5py
import numpy as np

# Load embeddings
with h5py.File('embeddings/00000358-159547_embeddings.hdf5', 'r') as f:
    bas_embeddings = f['BAS'][:]  # Shape: [num_epochs, 1, embed_dim]
    print(f"Embeddings shape: {bas_embeddings.shape}")
    print(f"Embedding dimension: {bas_embeddings.shape[-1]}")

# Load 5-minute aggregated embeddings
with h5py.File('embeddings/00000358-159547_embeddings_5min.hdf5', 'r') as f:
    bas_5min = f['BAS'][:]  # Shape: [num_5min_chunks, 1, embed_dim]
    print(f"5-min embeddings shape: {bas_5min.shape}")
```

## Next Steps

After generating embeddings, you can:
1. Use them for downstream tasks (sleep staging, disease prediction, etc.)
2. Fine-tune the model on your specific task
3. Analyze the embedding space
4. Train classifiers on the embeddings
