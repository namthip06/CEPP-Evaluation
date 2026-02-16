# Evaluation Script for mulEEG

## Overview

`evaluate_patients.py` loads a pretrained mulEEG model and evaluates it on preprocessed patient data, computing comprehensive performance metrics.

## Features

- ✅ **CPU-only inference** - No GPU required
- ✅ **Batch processing** - Evaluates all patients in output directory
- ✅ **Comprehensive metrics** - Accuracy, F1, Kappa, Balanced Accuracy
- ✅ **Per-class analysis** - Individual metrics for each sleep stage
- ✅ **Confusion matrix** - Visual performance analysis
- ✅ **Detailed reports** - CSV and text summaries

## Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Usage

### 1. Prepare Pretrained Weights

Place your pretrained model checkpoint at:
```
./saved_weights/ours_diverse_shhs.pt
```

### 2. Ensure Preprocessed Data Exists

Make sure you have preprocessed patient data in:
```
./output/
├── 00000358-159547/
│   └── 00000358-159547.npz
├── 00000359-159547/
│   └── 00000359-159547.npz
└── ...
```

### 3. Run Evaluation

```bash
python evaluate_patients.py
```

## Configuration

Edit the configuration in `main()` function:

```python
# Configuration
CHECKPOINT_PATH = "./saved_weights/ours_diverse_shhs.pt"  # Path to pretrained weights
OUTPUT_DIR = "./output"                                    # Directory with patient data
DEVICE = "cpu"                                             # Device (cpu/cuda)
```

## Output

The script generates three files in `./results/`:

### 1. `per_patient_results.csv`

Per-patient metrics in CSV format:

| patient_id | n_epochs | accuracy | macro_f1 | kappa | f1_wake | f1_n1 | f1_n2 | f1_n3 | f1_rem |
|------------|----------|----------|----------|-------|---------|-------|-------|-------|--------|
| 00000358-159547 | 900 | 0.8234 | 0.7891 | 0.7456 | 0.85 | 0.72 | 0.88 | 0.81 | 0.79 |

### 2. `evaluation_summary.txt`

Comprehensive text report including:
- Overall metrics across all patients
- Per-class precision, recall, F1-score
- Statistical summary of per-patient results

### 3. `confusion_matrix.png`

Visual confusion matrix showing:
- True vs predicted labels
- Per-class performance
- Misclassification patterns

## Metrics Explained

| Metric | Description | Range |
|--------|-------------|-------|
| **Accuracy** | Overall correct predictions | 0-1 |
| **Macro F1** | Average F1 across all classes (unweighted) | 0-1 |
| **Weighted F1** | Average F1 weighted by class support | 0-1 |
| **Cohen's Kappa** | Agreement beyond chance | -1 to 1 |
| **Balanced Accuracy** | Average recall per class | 0-1 |

## Sleep Stage Labels

| Label | Sleep Stage |
|-------|-------------|
| 0 | Wake |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 |
| 4 | REM |

## Example Output

```
======================================================================
mulEEG Sleep Stage Classification - Model Evaluation
======================================================================
Checkpoint: ./saved_weights/ours_diverse_shhs.pt
Data directory: ./output
Device: cpu
======================================================================
Loading model from: ./saved_weights/ours_diverse_shhs.pt
Model loaded successfully on cpu

Found 35 patient folders
======================================================================

Processing 1/35: 00000358-159547
  Epochs: 900
  Accuracy: 0.8234
  Macro F1: 0.7891
  Kappa: 0.7456

...

======================================================================
OVERALL EVALUATION RESULTS
======================================================================
Total Patients: 35
Total Epochs: 31,500
Overall Accuracy: 0.8123
Overall Macro F1: 0.7654
Overall Weighted F1: 0.8012
Overall Kappa: 0.7321
Overall Balanced Acc: 0.7543
======================================================================

Per-Class Performance:
              precision    recall  f1-score   support

        Wake     0.8456    0.8234    0.8344      5000
          N1     0.7123    0.6891    0.7005      3000
          N2     0.8789    0.8901    0.8845     12000
          N3     0.8234    0.8123    0.8178      7000
         REM     0.7891    0.8012    0.7951      4500

    accuracy                         0.8123     31500
   macro avg     0.8099    0.8032    0.7654     31500
weighted avg     0.8134    0.8123    0.8012     31500

Confusion matrix saved to: ./results/confusion_matrix.png
Per-patient results saved to: ./results/per_patient_results.csv
Evaluation summary saved to: ./results/evaluation_summary.txt

======================================================================
Evaluation completed successfully!
======================================================================
```

## Class Structure

### `PatientEvaluator`

Main evaluation class with methods:

- `__init__(checkpoint_path, output_dir, device)` - Initialize model and config
- `load_patient_data(patient_path)` - Load .npz file
- `predict_patient(x, batch_size)` - Run inference on patient data
- `evaluate_patient(patient_path)` - Compute metrics for one patient
- `evaluate_all_patients()` - Evaluate all patients and aggregate results
- `save_confusion_matrix(y_true, y_pred)` - Generate confusion matrix plot

## Troubleshooting

### Error: Checkpoint not found

```
Error: Checkpoint not found at ./saved_weights/ours_diverse_shhs.pt
```

**Solution:** Place pretrained weights in `./saved_weights/` directory

### Error: Output directory not found

```
Error: Output directory not found at ./output
```

**Solution:** Run preprocessing script first to generate patient data

### Memory Issues

If you encounter memory issues with large datasets:
- Reduce batch size in `predict_patient()` method
- Process patients in smaller groups

### Import Errors

If you get import errors:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Related Files

- [`preprocessing/custom/preprocess_custom.py`](file:///home/nummm/Documents/CEPP/mulEEG/preprocessing/custom/preprocess_custom.py) - Data preprocessing
- [`models/model.py`](file:///home/nummm/Documents/CEPP/mulEEG/models/model.py) - Model architecture
- [`config.py`](file:///home/nummm/Documents/CEPP/mulEEG/config.py) - Configuration
- [`helper_train.py`](file:///home/nummm/Documents/CEPP/mulEEG/helper_train.py) - Training utilities

## Notes

- Evaluation runs on CPU by default (no GPU required)
- Progress is printed for each patient during evaluation
- All results are saved automatically to `./results/` directory
- The script handles missing or corrupted patient data gracefully
