# L-SeqSleepNet Custom Pipeline

This directory contains scripts to run the **L-SeqSleepNet** model on custom data using the **SleepEDF-20** pretrained weights. The pipeline handles preprocessing, inference, evaluation, and visualization.

## Folder Structure

The scripts expect the following folder structure:

```
/home/nummm/Documents/CEPP/
├── rawEEG/                     # Input directory
│   ├── [subject_id]/
│   │   ├── edf_signals.edf     # Raw EEG data (EDF format)
│   │   └── csv_hypnogram.csv   # Sleep staging labels (CSV)
│   └── ...
└── l-seqsleepnet/
    └── custom/                 # This directory
        ├── run_inference.py    # Main script (Preprocessing + Inference)
        ├── evaluate_custom.py  # Evaluation script
        ├── preprocessing_output/ # Generated .mat files
        ├── results/            # Generated inference results
        ├── evaluation/         # Generated metrics and plots
        └── visualization_results.ipynb
```

---

## End-to-End Workflow

### Step 1: Prepare Data
Ensure your raw data is in `rawEEG/`.
- Each subject must have their own folder.
- Inside each folder, there must be:
    - `edf_signals.edf`: The EDF file containing EEG signals.
    - `csv_hypnogram.csv`: A CSV file with sleep stages (Wake, N1, N2, N3, REM).

### Step 2: Run Inference `run_inference.py`
This is the main script. It performs two phases:
1.  **Preprocessing**: Converts EDFs to spectrograms (`.mat` files) in `preprocessing_output/`.
2.  **Inference**: Loads the pretrained L-SeqSleepNet model and generates predictions in `results/`.

**Usage:**
```bash
python3 run_inference.py
```
*Note: If data is already preprocessed, it skips Phase 1 automatically.*

### Step 3: Evaluate Results `evaluate_custom.py`
Calculates performance metrics (Accuracy, F1-Score, Cohen's Kappa, etc.) by comparing predictions against the ground truth from the hypnograms.

**Usage:**
```bash
python3 evaluate_custom.py
```
**Output (`custom/evaluation/`):**
- `overall_metrics.json`: Aggregated metrics across all subjects.
- `per_subject_metrics.csv`: Detailed metrics for each subject.
- `confusion_matrix.csv`: Confusion matrix of the predictions.

### Step 4: Visualization `visualization_results.ipynb`
Open this Jupyter Notebook to visualize the evaluation results. It generates:
- Summary bar charts.
- Confusion matrix heatmaps.
- Distribution of accuracy per subject.
- F1-scores per sleep stage.

**Usage:**
Open `visualization_results.ipynb` in Jupyter Lab or VS Code and run all cells.

---

## Advanced / Debugging

### `preprocess_rawEEG.py`
This is a standalone version of the preprocessing logic found in `run_inference.py`. You can use this if you only want to generate spectrograms without loading the model.
```bash
python3 preprocess_rawEEG.py
```

## Requirements & Notes
- **Low RAM Mode**: scripts process subjects sequentially to avoid OOM errors.
- **Absolute Paths**: The scripts use absolute paths rooted at `/home/nummm/Documents/CEPP/`. If you move the project, update `BASE_DIR` or `RAW_EEG_DIR` in the scripts.
- **Model Constraints**:
    - Sampling Rate: 100 Hz
    - Channels: 1 EEG channel (Fpz-Cz or similar)
    - Epoch: 30 seconds
