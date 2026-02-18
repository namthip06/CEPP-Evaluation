# Sleep Staging Prediction Guide

## Purpose

This guide explains how to use the `predict_sleep_stages.py` script. The goal of this script is to take the "summaries" (embeddings) of the brain waves and use AI to decide **what sleep stage the patient is in** for every 30 seconds.

It predicts 5 distinct stages:
*   **Wake** (0)
*   **Stage 1** (1) - Light sleep
*   **Stage 2** (2) - Light sleep
*   **Stage 3** (3) - Deep sleep
*   **REM** (4) - Dreaming sleep

## Prerequisites

Before running this script, you must have:
1.  **Generated Embeddings**: Ran `generate_embeddings.py` successfully.
2.  **Ground Truth Labels**: Have the original label CSV files (usually from `preprocess.py`).

## Folder Structure

The script relies on this structure:

```
/home/nummm/Documents/CEPP/sleepfm-clinical/custom/embeddings/
├── 00000358-159547_embeddings.hdf5      <-- INPUT: Brain wave summaries

/home/nummm/Documents/CEPP/rawEEG/
├── 00000358-159547/
│   └── 00000358-159547.csv              <-- INPUT: Correct answers (Labels)
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
python predict_sleep_stages.py
```

### 3. What Happens Next?
The script will:
*   **Match Files**: Find the embedding file and its corresponding label file.
*   **Think**: Fed the data into the model to makes a guess for every 30-second chunk.
*   **Compare**: Check if its guess matches the doctor's label (Ground Truth).
*   **Save**: Write down its guesses and how confident it felt about them.

### 4. Output

The predictions are saved here:

```
/home/nummm/Documents/CEPP/sleepfm-clinical/custom/predictions/
```

For each patient, you get a CSV file like `00000358-159547_predictions.csv`:

| Epoch | GroundTruth | Predicted | Confidence_Wake | Confidence_REM | ... |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 2 (Stage 2) | 2 (Stage 2) | 0.05 | 0.02 | ... |
| 1 | 3 (Deep) | 3 (Deep) | 0.01 | 0.01 | ... |
| ... | ... | ... | ... | ... | ... |

*   **Epoch**: Time step (0 = first 30s, 1 = second 30s, etc.)
*   **GroundTruth**: What the doctor said.
*   **Predicted**: What the AI said.
*   **Confidence**: How sure the AI was (0.0 to 1.0).

## Evaluating Results

To see how well the AI did overall (Accuracy, Confusion Matrix, etc.), you can run the verification script afterwards:

```bash
python verify_predictions.py
```

This will produce summary files in the `predictions/` folder telling you the overall accuracy across all patients.

## Troubleshooting

### "Embeddings directory not found"
*   **Fix**: Run `generate_embeddings.py` first.

### "Label file not found"
*   **Fix**: Run `preprocess.py` first to generate the label CSVs.

### Low Accuracy?
*   Sleep staging is hard! 70-80% agreement with humans is considered good.
*   Check if your labels (hypnograms) are reliable.
