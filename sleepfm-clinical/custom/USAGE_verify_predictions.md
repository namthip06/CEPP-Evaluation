# Prediction Verification Guide

## Purpose

This guide explains how to use the `verify_predictions.py` script. The goal of this script is to **"grade" the AI's performance**.

After the AI has made its predictions (using `predict_sleep_stages.py`), this script compares those predictions against the correct answers (Ground Truth) to see how well it did.

## Prerequisites

Before running this script, you must have:
1.  **Generated Predictions**: Ran `predict_sleep_stages.py` successfully.
    *   This ensures that prediction files exist in `/custom/predictions/`.

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
python verify_predictions.py
```

### 3. What Happens Next?
The script will:
*   **Gather Files**: Collect all prediction files (`.csv`) from the predictions folder.
*   **Calculate Scores**: Compute accuracy and other metrics for *all patients combined*.
*   **Generate Report**: Print a summary to the screen and save detailed reports to files.

## Outputs

The evaluation results are saved in:

```
/home/nummm/Documents/CEPP/sleepfm-clinical/custom/predictions/
```

The script creates three files to help you analyze performance from different angles:

### 1. Overall Metrics (`evaluation_metrics_overall.csv`)
Gives you the "biq picture" numbers.
*   **Accuracy**: The percentage of 30-second chunks the AI got exactly right.
*   **Cohen's Kappa**: A more strict score that accounts for lucky guesses.

### 2. Per-Class Metrics (`evaluation_metrics_per_class.csv`)
Shows how well the AI does on *specific* sleep stages.
*   **Precision**: When the AI predicts "REM", how often is it actually REM?
*   **Recall**: Out of all the actual "REM" sleep, how much did the AI find?
*   **F1-Score**: A balanced score combining Precision and Recall.

### 3. Confusion Matrix (`evaluation_metrics_confusion_matrix.csv`)
A table showing exactly where the AI is making mistakes.
*   Rows = Actual Sleep Stages (Ground Truth)
*   Columns = Predicted Sleep Stages

**Example:**
If the row is "Stage 2" and the column "Stage 3" has a high number, it means the AI often confuses Stage 2 for Stage 3.

## Metric Definitions

Here is a simple cheat sheet for the metrics:

| Metric | Question it Answers | Good for... |
| :--- | :--- | :--- |
| **Accuracy** | "How often is the AI right?" | General performance check. |
| **Precision** | "When it claims X, can I trust it?" | Minimal false alarms (e.g., classifying text). |
| **Recall** | "Did it find all instances of X?" | Critical detection (e.g., finding diseases in sleep). |
| **F1-Score** | "Is it balanced?" | Best overall measure if classes are uneven (e.g., rare sleep stages). |
