# Sleep Staging Prediction Guide

## Overview
This guide explains how to predict sleep stages (Wake, Stage 1, Stage 2, Stage 3, REM) from raw EEG data using the pretrained SleepFM sleep staging model.

## Prerequisites

### 1. Generated Embeddings
Before predicting sleep stages, you must first generate embeddings using `generate_embeddings.py`:

```bash
cd /home/nummm/Documents/CEPP/sleepfm-clinical/custom
python generate_embeddings.py
```

This will create embedding files in `/custom/embeddings/`.

### 2. Required Files
Each subject should have:
- **Embeddings**: `/custom/embeddings/{subject_id}_embeddings.hdf5`
- **Ground truth labels**: `/home/nummm/Documents/CEPP/rawEEG/{subject_id}/{subject_id}.csv`

## Usage

### Predict Sleep Stages

```bash
cd /home/nummm/Documents/CEPP/sleepfm-clinical/custom
python predict_sleep_stages.py
```

### What It Does

1. **Loads the pretrained model**: Uses `model_sleep_staging` checkpoint
2. **Processes each subject sequentially**: One at a time to avoid RAM issues
3. **Loads embeddings**: From `/custom/embeddings/`
4. **Loads ground truth labels**: From the raw EEG directory
5. **Generates predictions**: For each 30-second epoch
6. **Saves predictions**: Creates CSV files with predictions and confidence scores

### Output

Predictions are saved to:
```
/home/nummm/Documents/CEPP/sleepfm-clinical/custom/predictions/
```

Each subject will have a prediction file:
- `{subject_id}_predictions.csv`

**CSV Format**:
```csv
Epoch,GroundTruth,Predicted,Confidence_Wake,Confidence_Stage1,Confidence_Stage2,Confidence_Stage3,Confidence_REM
0,2,2,0.05,0.10,0.75,0.08,0.02
1,2,2,0.03,0.12,0.78,0.05,0.02
2,3,3,0.02,0.05,0.15,0.75,0.03
...
```

**Columns**:
- `Epoch`: Epoch number (30-second intervals)
- `GroundTruth`: True sleep stage (0=Wake, 1=Stage 1, 2=Stage 2, 3=Stage 3, 4=REM)
- `Predicted`: Predicted sleep stage
- `Confidence_*`: Probability for each sleep stage (sums to 1.0)

## Evaluating Predictions

### Calculate Metrics

After generating predictions, evaluate the results:

```bash
python verify_predictions.py
```

This will calculate:
- **Overall accuracy**: Percentage of correctly predicted epochs
- **Per-class metrics**: Precision, recall, F1-score for each sleep stage
- **Confusion matrix**: Shows which stages are confused with each other
- **Cohen's Kappa**: Agreement measure accounting for chance

### Output Files

The evaluation creates three CSV files in `/custom/predictions/`:

1. **`evaluation_metrics_overall.csv`**: Overall metrics
   ```csv
   Metric,Value
   Accuracy,0.78
   Cohen's Kappa,0.72
   Precision (Macro),0.75
   ...
   ```

2. **`evaluation_metrics_per_class.csv`**: Per-class metrics
   ```csv
   Class,Precision,Recall,F1-Score,Support
   Wake,0.85,0.82,0.83,15234
   Stage 1,0.45,0.38,0.41,3421
   ...
   ```

3. **`evaluation_metrics_confusion_matrix.csv`**: Confusion matrix
   ```csv
   ,Wake,Stage 1,Stage 2,Stage 3,REM
   Wake,12500,234,1200,50,1250
   Stage 1,456,1300,1200,234,231
   ...
   ```

## Configuration

The script uses these default settings:
- **Model**: `../sleepfm/checkpoints/model_sleep_staging`
- **Embeddings directory**: `/home/nummm/Documents/CEPP/sleepfm-clinical/custom/embeddings`
- **Raw EEG directory**: `/home/nummm/Documents/CEPP/rawEEG`
- **Output directory**: `/home/nummm/Documents/CEPP/sleepfm-clinical/custom/predictions`
- **Batch size**: 1 (sequential processing)
- **Device**: CUDA with CPU fallback

To modify these settings, edit the `main()` function in `predict_sleep_stages.py`.

## Sleep Stage Mapping

The model predicts 5 sleep stages:

| Stage Number | Stage Name | Description |
|--------------|------------|-------------|
| 0 | Wake | Awake |
| 1 | Stage 1 | Light sleep (N1) |
| 2 | Stage 2 | Light sleep (N2) |
| 3 | Stage 3 | Deep sleep (N3/SWS) |
| 4 | REM | Rapid Eye Movement sleep |

## Memory Optimization

The script is designed to minimize RAM usage:
- **Sequential processing**: Processes one subject at a time
- **No batching**: Uses batch_size=1
- **CUDA cache clearing**: Clears GPU memory after each subject
- **Efficient loading**: Loads only necessary data

## Troubleshooting

### "Embeddings directory not found"
- Make sure you ran `generate_embeddings.py` first
- Check that embeddings are in `/custom/embeddings/`

### "Label file not found"
- Ensure you ran `preprocess.py` to create label CSV files
- Check that labels exist in `/home/nummm/Documents/CEPP/rawEEG/{id}/{id}.csv`

### Out of Memory Error
- The script already uses minimal RAM settings
- Try closing other applications
- Consider processing fewer subjects at a time

### Low Accuracy
- Sleep staging is challenging; 70-80% accuracy is typical
- Check that embeddings were generated correctly
- Verify that ground truth labels are correct
- Some datasets may have different sleep stage definitions

## Reading Predictions in Python

```python
import pandas as pd
import numpy as np

# Load predictions
df = pd.read_csv('predictions/00000358-159547_predictions.csv')

# Get predictions and ground truth
predictions = df['Predicted'].values
ground_truth = df['GroundTruth'].values

# Calculate accuracy
accuracy = (predictions == ground_truth).mean()
print(f"Accuracy: {accuracy*100:.2f}%")

# Get confidence scores
confidences = df[['Confidence_Wake', 'Confidence_Stage1', 
                  'Confidence_Stage2', 'Confidence_Stage3', 
                  'Confidence_REM']].values

# Find epochs with low confidence
max_confidence = confidences.max(axis=1)
low_confidence_epochs = np.where(max_confidence < 0.5)[0]
print(f"Low confidence epochs: {len(low_confidence_epochs)}")
```

## Next Steps

After generating predictions, you can:
1. **Analyze sleep architecture**: Calculate sleep efficiency, sleep latency, etc.
2. **Visualize hypnograms**: Plot predicted vs. ground truth sleep stages
3. **Error analysis**: Identify which stages are most often confused
4. **Clinical applications**: Use predictions for sleep disorder detection
