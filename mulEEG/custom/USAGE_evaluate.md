# evaluate.py Usage Guide

## Purpose
The `evaluate.py` script evaluates the performance of the pre-trained `mulEEG` sleep stage classification model on your custom dataset.

It performs the following:
1.  Loads a pre-trained model checkpoint (`weights/shhs/ours_diverse.pt`).
2.  Loads preprocessed patient data (`.npz` files) from `custom/preprocessing_output/`.
3.  Runs inference on each patient's data using the CPU (optimized for low-RAM environments).
4.  Calculates performance metrics (Accuracy, F1-Score, Cohen's Kappa, Balanced Accuracy).
5.  Generates a confusion matrix and detailed classification reports.

## Prerequisites
Before running this script, ensure you have:
1.  **Preprocessed Data**: You must run `preprocess.py` first. The output files should be in `custom/preprocessing_output/`.
2.  **Model Checkpoint**: The script expects the pre-trained model weights at `weights/shhs/ours_diverse.pt`.
    *   *Note: If your checkpoint is elsewhere, update the `CHECKPOINT_PATH` variable in the script.*

## Configuration
Key configurations are found in the `main()` function of `evaluate.py`.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `checkpoint_path` | `.../weights/shhs/ours_diverse.pt` | Path to the trained model file. |
| `preprocessing_out_dir` | `.../custom/preprocessing_output` | Directory containing `.npz` patient files. |
| `results_dir` | `.../custom/results` | Directory where results will be saved. |
| `device` | `'cpu'` | Processing device. Defaults to `cpu` to avoid OOM errors on standard machines. |

## Usage
Run the script from the terminal:

```bash
# Navigate to the custom directory
cd /home/nummm/Documents/CEPP/mulEEG/custom

# Run the evaluation
python evaluate.py
```

## Output
Results are saved to `mulEEG/custom/results/`:

*   **`per_patient_results.csv`**: A CSV file containing metrics (Accuracy, F1, Kappa) for every individual patient.
*   **`evaluation_summary.txt`**: A text file summarizing overall performance, including global averages and per-class metrics.
*   **`confusion_matrix.png`**: A visualization of the confusion matrix showing predicted vs. actual sleep stages.

## Metrics
The evaluation reports:
*   **Overall Accuracy**: Percentage of correct predictions.
*   **Macro F1**: Average F1-score across all classes (treats all classes equally).
*   **Weighted F1**: F1-score weighted by class prevalence.
*   **Cohen's Kappa**: Agreement between model and ground truth, accounting for chance.
*   **Balanced Accuracy**: Average of recall obtained on each class.
