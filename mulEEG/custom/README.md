# mulEEG Custom Pipeline Guide

This guide explains how to preprocess raw continuous EEG data and evaluate it using the pre-trained **mulEEG** model.

The workflow consists of two main steps:
1.  **Preprocessing**: Converting raw EDF/CSV files into model-ready `.npz` files.
2.  **Evaluation**: Running the model on the processed data and analyzing performance.

---

## 1. Prerequisites & Folder Structure

To run the pipeline, ensure your directories are structured as follows:

```text
/Documents/CEPP/
├── mulEEG/                  <-- Project Root
│   └── custom/
│       ├── preprocess.py    <-- Script 1: Preprocessing
│       └── evaluate.py      <-- Script 2: Evaluation
└── rawEEG/                  <-- Input Data (Sibling of mulEEG)
    ├── [patient_id_1]/
    │   ├── edf_signals.edf      <-- Required: Raw Signals
    │   └── csv_hypnogram.csv    <-- Required: Sleep Stages
    └── [patient_id_2]/
        ...
```

> **Note**: The `rawEEG` folder must be located one level up from `mulEEG/custom/`, alongside the `mulEEG` folder itself.

---

## 2. Step 1: Preprocessing (`preprocess.py`)

This script reads raw EDF files and CSV hypnograms, standardizes them, and saves them as `.npz` files.

### What it does:
*   Reads EDF signals and downsamples to 100 Hz.
*   Maps sleep stages (WK, N1, N2, N3, REM) to numeric labels (0-4).
*   Segments data into 30-second epochs.
*   Trims excessive wake periods from the start and end.

### How to Run:
Open your terminal and run:

```bash
cd /home/nummm/Documents/CEPP/mulEEG/custom
python preprocess.py
```

### Configuration:
You can adjust settings directly in `preprocess.py` (inside the `main()` function):
*   `select_channel`: Specific EEG channel to use (default: `None` = auto-select).
*   `trim_wake_edges`: Set to `True` to remove long wake periods at the start/end.
*   `epoch_sec_size`: Duration of each epoch (default: `30` seconds).

### Output:
Processed files are saved to `mulEEG/custom/preprocessing_output/`.

---

## 3. Step 2: Evaluation (`evaluate.py`)

This script loads the preprocessed data, runs the pre-trained model, and calculates performance metrics.

### What it does:
*   Loads the pre-trained model (`weights/shhs/ours_diverse.pt`).
*   Runs inference on each patient's data (CPU-optimized for low RAM).
*   Generates accuracy, F1-scores, and confusion matrices.

### How to Run:
Once preprocessing is complete, run:

```bash
cd /home/nummm/Documents/CEPP/mulEEG/custom
python evaluate.py
```

### Configuration:
Settings in `evaluate.py`:
*   `checkpoint_path`: Path to model weights.
*   `device`: Set to `"cpu"` by default for stability.

---

## 4. Understanding the Results

After running `evaluate.py`, check the `mulEEG/custom/results/` folder for:

1.  **`evaluation_summary.txt`**:
    *   **Overall Metrics**: Accuracy, Macro F1, Kappa across all patients.
    *   **Per-Class Performance**: Precision/Recall for Wake, N1, N2, N3, REM.
    *   **Patient Stats**: Min/Max/Mean accuracy per patient.

2.  **`per_patient_results.csv`**:
    *   Detailed metrics for *each* individual patient. Use this to identify outliers or specific patients where the model underperforms.

3.  **`confusion_matrix.png`**:
    *   Visualizes where the model makes mistakes (e.g., confusing N1 with Wake).

---

### Troubleshooting

*   **"Input directory does not exist"**: Ensure `rawEEG` is in the correct location (see Section 1).
*   **"Checkpoint not found"**: Verify you have `weights/shhs/ours_diverse.pt`.
*   **"Memory Error"**: The scripts are optimized for CPU, but ensure you have at least 4-8GB RAM available.
