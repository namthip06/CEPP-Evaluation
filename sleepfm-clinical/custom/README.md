# SleepFM Clinical Pipeline

This folder contains the custom scripts for the SleepFM clinical pipeline. The pipeline is designed to process raw PSG data, extract embeddings using the pre-trained SleepFM model, predict sleep stages, and verify the results.

## Pipeline Overview

The pipeline consists of 4 main steps:

```mermaid
graph TD
    A[Raw Data (EDF)] -->|1. Preprocess| B(HDF5 + Labels)
    B -->|2. Generate Embeddings| C(Embeddings)
    C -->|3. Predict| D(Predictions CSV)
    D -->|4. Verify| E(Evaluation Report)
```

## Step-by-Step Guide

### 1. Preprocessing (`preprocess.py`)
Converts raw EDF files into the optimized HDF5 format and extracts sleep stage labels.

*   **Input**: Raw EDF files in `rawEEG/`
*   **Output**: `.hdf5` and `.csv` files in `rawEEG/`
*   **Command**: `python preprocess.py`
*   **Documentation**: [USAGE_preprocess.md](USAGE_preprocess.md)

### 2. Generate Embeddings (`generate_embeddings.py`)
Extracts "brain wave summaries" (embeddings) from the preprocessed data using the SleepFM model.

*   **Input**: `.hdf5` files from Step 1
*   **Output**: Embedding files in `custom/embeddings/`
*   **Command**: `python generate_embeddings.py`
*   **Documentation**: [USAGE_generate_embeddings.md](USAGE_generate_embeddings.md)

### 3. Predict Sleep Stages (`predict_sleep_stages.py`)
Uses the embeddings to predict sleep stages (Wake, Light, Deep, REM) for every 30-second epoch.

*   **Input**: Embeddings from Step 2
*   **Output**: Prediction CSVs in `custom/predictions/`
*   **Command**: `python predict_sleep_stages.py`
*   **Documentation**: [USAGE_predict_sleep_stages.md](USAGE_predict_sleep_stages.md)

### 4. Verify Predictions (`verify_predictions.py`)
Compares the AI's predictions against the doctor's ground truth labels to calculate accuracy and other metrics.

*   **Input**: Prediction CSVs from Step 3
*   **Output**: Evaluation report in `custom/predictions/`
*   **Command**: `python verify_predictions.py`
*   **Documentation**: [USAGE_verify_predictions.md](USAGE_verify_predictions.md)

## Quick Start

To run the full pipeline:

```bash
# Activate environment
conda activate sleepfm

# 1. Preprocess data
python preprocess.py

# 2. Extract features
python generate_embeddings.py

# 3. Predict sleep stages
python predict_sleep_stages.py

# 4. Check results
python verify_predictions.py
```

## Folder Structure

*   `embeddings/`: Stores the generated intermediate features.
*   `predictions/`: Stores the final outputs and evaluation reports.
*   `USAGE_*.md`: Detailed guides for each script.
