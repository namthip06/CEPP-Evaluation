# Embedding Generation Guide

## Purpose

This guide explains how to use the `generate_embeddings.py` script. The goal of this script is to extract **features** (called "embeddings") from your raw sleep data using the pre-trained SleepFM model.

Think of embeddings as a "summary" of the brain waves (EEG) that a computer can understand. These summaries are used for:
1.  Sleep staging (classifying sleep stages like Rem, Deep Sleep).
2.  Disease prediction (finding patterns related to disorders).
3.  Other analysis tasks.

## Prerequisites

Before running this script, you must have:
1.  **Preprocessed Data**: Ran `preprocess.py` to create `.hdf5` files.
2.  **Model**: Have the SleepFM model checkpoint at `../sleepfm/checkpoints/model_base`.

## Folder Structure

The script expects your data to look like this (created by `preprocess.py`):

```
/home/nummm/Documents/CEPP/rawEEG/
├── 00000358-159547/
│   ├── ... (original files)
│   ├── 00000358-159547.hdf5      <-- REQUIRED INPUT
│   └── 00000358-159547.csv       <-- LABEL INPUT
└── ...
```

## Usage

### 1. Setup Environment
Open your terminal and activate the environment:

```bash
conda activate sleepfm
```

### 2. Run the Script
Navigate to the custom folder and run the script:

```bash
cd /home/nummm/Documents/CEPP/sleepfm-clinical/custom
python generate_embeddings.py
```

### 3. What Happens Next?
The script will:
*   **Load the SleepFM Model**: It uses the pre-trained model to "read" the EEG data.
*   **Process One by One**: It goes through each patient folder one at a time to save memory.
*   **Extract Features**: It specifically looks at the **Brain Activity Signals (EEG)**.
*   **Save Results**: It saves the extracted features into a new folder.

**Note:** By default, this script runs on **CPU** to ensure compatibility.

### 4. Output

The embeddings (features) are saved here:

```
/home/nummm/Documents/CEPP/sleepfm-clinical/custom/embeddings/
```

For each patient, you get two files:
1.  **`[id]_embeddings.hdf5`**: Features for every 30-second epoch (Detailed).
2.  **`[id]_embeddings_5min.hdf5`**: Features aggregated every 5 minutes (Summary).

## Configuration

You can change settings in the `main()` function of `generate_embeddings.py`:

*   **`raw_eeg_dir`**: Where your raw data folders are.
*   **`output_dir`**: Where to save the embeddings.
*   **`device`**: Changed to `"cuda"` if you have a GPU and enough memory (in `EmbeddingGenerator.__init__`).

## Troubleshooting

### "HDF5 file not found"
*   **Cause**: You haven't run `preprocess.py` yet, or it failed for that folder.
*   **Fix**: Run `python preprocess.py` first.

### "Out of Memory"
*   **Cause**: The model is too big for your RAM.
*   **Fix**: The script is already optimized (batch size = 1). Try closing other programs.

### Speed
*   **Note**: Since it runs on CPU, it might take some time per patient. This is normal.
