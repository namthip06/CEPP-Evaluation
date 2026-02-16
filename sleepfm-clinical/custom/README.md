# SleepFM Demo: Running Inference with EEG Files

This directory contains a complete end-to-end demonstration of how to use SleepFM for sleep analysis and disease prediction using polysomnography (PSG) data.

## 📋 Overview

The demo notebook (`demo.ipynb`) demonstrates the complete SleepFM pipeline:

1. **Preprocessing**: Converting EDF files to HDF5 format
2. **Embedding Generation**: Extracting sleep representations using the pretrained SleepFM model
3. **Sleep Staging**: Predicting sleep stages (Wake, Stage 1, Stage 2, Stage 3, REM)
4. **Disease Prediction**: Predicting risk for 1,065 different medical conditions

## 🚀 Quick Start

### Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended: NVIDIA A40/A100, minimum: RTX 2080 Ti)
- At least 32 GB RAM
- SleepFM environment set up (see main README.md)

### Required Files

Before running the demo, ensure you have:

1. **EDF file**: Your polysomnography recording in EDF format
2. **Sleep stage labels** (for sleep staging): CSV file with columns:
   - `Start`, `Stop`, `StageName`, `StageNumber`
3. **Demographics** (for disease prediction): CSV file with:
   - `Study ID`, age, gender, and other demographic features

## 📁 Demo Data Structure

The demo uses the following data structure:

```
custom/
├── demo.ipynb                    # Main demo notebook
└── demo_data/                    # Generated during demo execution
    ├── demo_psg.edf              # Input: Raw PSG recording
    ├── demo_psg.hdf5             # Preprocessed PSG data
    ├── demo_psg.csv              # Sleep stage labels
    ├── demo_age_gender.csv       # Demographics for disease prediction
    ├── demo_emb/                 # 5-second granular embeddings
    │   └── demo_psg.hdf5
    ├── demo_5min_agg_emb/        # 5-minute aggregated embeddings
    │   └── demo_psg.hdf5
    ├── demo_sleep_staging/       # Sleep staging results
    │   ├── all_targets.pickle
    │   ├── all_outputs.pickle
    │   ├── all_logits.pickle
    │   ├── all_masks.pickle
    │   └── all_paths.pickle
    └── demo_diagnosis/           # Disease prediction results
        ├── all_outputs.pickle
        ├── all_event_times.pickle
        ├── all_is_event.pickle
        └── all_paths.pickle
```

## 🔧 Step-by-Step Guide

### Part 0: Preprocessing EDF Files

The first step converts your raw EDF file to HDF5 format:

```python
from preprocessing.preprocessing import EDFToHDF5Converter

# Initialize converter
converter = EDFToHDF5Converter(
    root_dir="/edf_root",      # Dummy root for single file
    target_dir="/note",         # Dummy target for single file
    resample_rate=128           # Resample to 128 Hz
)

# Convert single file
edf_path = "demo_data/demo_psg.edf"
hdf5_path = "demo_data/demo_psg.hdf5"
converter.convert(edf_path, hdf5_path)
```

**Key Parameters:**
- `resample_rate`: Target sampling frequency (default: 128 Hz)
- The converter automatically handles channel mapping based on `sleepfm/configs/channel_groups.json`

**Important Note:** Before preprocessing, review the channel mappings in `sleepfm/configs/channel_groups.json` to ensure all channels in your PSG data are correctly categorized into modalities (BAS, RESP, EKG, EMG).

### Part 1: Generating Embeddings

Extract sleep representations using the pretrained SleepFM model:

```python
# Load pretrained model
model_path = "../sleepfm/checkpoints/model_base"
config = load_config(os.path.join(model_path, "config.json"))

# Initialize model
model = SetTransformer(
    in_channels=config["in_channels"],
    patch_size=config["patch_size"],
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    pooling_head=config["pooling_head"],
    dropout=0.0
)

# Load weights
checkpoint = torch.load(os.path.join(model_path, "best.pt"))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Generate embeddings
# This produces two types of embeddings:
# 1. Granular 5-second embeddings (demo_emb/)
# 2. Aggregated 5-minute embeddings (demo_5min_agg_emb/)
```

**Model Architecture:**
- **Input**: Multi-modal PSG signals (BAS, RESP, EKG, EMG)
- **Output**: 128-dimensional embeddings
- **Parameters**: 4.44 million trainable parameters
- **Modalities**: 
  - BAS (Brain Activity Signals): EEG channels
  - RESP (Respiratory): Airflow, effort signals
  - EKG (Cardiac): ECG signals
  - EMG (Muscle Activity): Chin, leg EMG

### Part 2: Sleep Staging

Predict sleep stages using the finetuned sleep staging model:

```python
# Load sleep staging model
sleep_staging_model_path = "../sleepfm/checkpoints/model_sleep_staging"
config = load_data(os.path.join(sleep_staging_model_path, "config.json"))

# Initialize and load model
model = SleepEventLSTMClassifier(**config['model_params'])
checkpoint = torch.load(os.path.join(sleep_staging_model_path, "best.pth"))
model.load_state_dict(checkpoint)

# Prepare dataset with labels
hdf5_paths = ["demo_data/demo_emb/demo_psg.hdf5"]
label_files = ["demo_data/demo_psg.csv"]
dataset = SleepEventClassificationDataset(
    config, channel_groups, 
    hdf5_paths=hdf5_paths, 
    label_files=label_files,
    split="test"
)

# Run inference
# Output: Predictions for 5 sleep stages
# - Wake (0)
# - Stage 1 (1)
# - Stage 2 (2)
# - Stage 3 (3)
# - REM (4)
```

**Sleep Stage Labels Format:**
Your CSV file should have the following structure:
```csv
Start,Stop,StageName,StageNumber
0.0,30.0,Wake,0
30.0,60.0,Stage 1,1
60.0,90.0,Stage 2,2
```

**Evaluation Metrics:**
- Per-class F1 scores
- Confusion matrix
- Overall accuracy

### Part 3: Disease Prediction

Predict disease risk using the Cox Proportional Hazards model:

```python
# Load disease prediction model
disease_model_path = "../sleepfm/checkpoints/model_diagnosis"
config = load_data(os.path.join(disease_model_path, "config.json"))

# Initialize model
model = DiagnosisFinetuneFullLSTMCOXPHWithDemo(**config['model_params'])
checkpoint = torch.load(os.path.join(disease_model_path, "best.pth"))
model.load_state_dict(checkpoint)

# Prepare dataset with demographics
hdf5_paths = ["demo_data/demo_emb/demo_psg.hdf5"]
demo_labels_path = "demo_data/demo_age_gender.csv"
dataset = DiagnosisFinetuneFullCOXPHWithDemoDataset(
    config, channel_groups,
    hdf5_paths=hdf5_paths,
    demo_labels_path=demo_labels_path,
    split="test"
)

# Run inference
# Output: Hazard predictions for 1,065 conditions
```

**Demographics File Format:**
```csv
Study ID,Age,Gender,BMI,...
demo_psg,45,1,28.5,...
```

**Disease Mapping:**
The model outputs predictions for 1,065 conditions. Map predictions to disease names using:
```python
labels_df = pd.read_csv("../sleepfm/configs/label_mapping.csv")
labels_df["output"] = all_outputs[0]
labels_df["is_event"] = all_is_event[0]
labels_df["event_time"] = all_event_times[0]
```

**Output Interpretation:**
- `output`: Log hazard ratio (higher = higher risk)
- `is_event`: Whether the event occurred (0/1)
- `event_time`: Time to event in days

## 📊 Expected Results

### Sleep Staging Performance
On real data (not synthetic demo data), SleepFM achieves:
- Mean F1 score: 0.70–0.78 across sleep stages
- Competitive with specialized models (U-Sleep, YASA)

### Disease Prediction Performance
SleepFM achieves C-Index ≥ 0.75 for 130 conditions, including:
- All-cause mortality: 0.84
- Dementia: 0.85
- Myocardial infarction: 0.81
- Heart failure: 0.80
- Chronic kidney disease: 0.79
- Stroke: 0.78
- Atrial fibrillation: 0.78

## ⚠️ Important Notes

### Synthetic Demo Data
**All data in this demo is synthetically generated** and is for demonstration purposes only. The sleep stage annotations, demographics, and disease labels are not real clinical data.

### Channel Mapping
Before processing your own data, **review and update** `sleepfm/configs/channel_groups.json` to ensure all channels in your PSG recordings are correctly mapped to modalities. Consult with domain experts to categorize dataset-specific channels appropriately.

### Computational Requirements
- **Preprocessing**: ~1 minute per hour of PSG data
- **Embedding generation**: ~8 seconds per file (on A40 GPU)
- **Sleep staging**: ~1 second per file
- **Disease prediction**: ~1 second per file

### Batch Processing
For processing multiple files, use the scripts in `sleepfm/preprocessing/` and `sleepfm/pipeline/`:
- `preprocessing.sh`: Batch EDF to HDF5 conversion
- `generate_embeddings.py`: Batch embedding generation
- `evaluate_sleep_staging.py`: Batch sleep staging evaluation

## 🔍 Troubleshooting

### Common Issues

1. **Missing channels in HDF5 file**
   - Check that your EDF file contains the expected channels
   - Update `channel_groups.json` to include your specific channels

2. **CUDA out of memory**
   - Reduce batch size in the config
   - Use smaller GPU-compatible batch sizes

3. **Label mismatch errors**
   - Ensure CSV filenames match HDF5 filenames (without extension)
   - Verify CSV format matches expected structure

4. **Embedding dimension mismatch**
   - Ensure you're using embeddings from the correct directory
   - Check that preprocessing used the same config as the model

## 📚 Additional Resources

- **Main README**: `../README.md` - Full installation and usage guide
- **Channel Groups**: `../sleepfm/configs/channel_groups.json` - Modality mappings
- **Label Mapping**: `../sleepfm/configs/label_mapping.csv` - Disease phecode mappings
- **Paper**: https://doi.org/10.1038/s41591-025-04133-4

## 📧 Support

For questions or issues:
1. Check the main repository README
2. Review the paper for methodological details
3. Open an issue on the GitHub repository

## 📄 License

MIT License - See LICENSE file in the main repository
