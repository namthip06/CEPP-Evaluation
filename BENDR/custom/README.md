# BENDR Custom Pipeline

This repository contains a custom pipeline for raw EEG data preprocessing and sequence prediction using the **BENDR** (BendingCollegeWav2Vec) architecture. 

## System Requirements
- Python (via Virtual Environment such as `.venv` or `uv`)
- Dependencies: `mne`, `numpy`, `pandas`, `torch`, `tqdm`, `psutil`, `openpyxl`

---

## 1. Preprocessing (`preprocess_rawEEG.py`)

This script handles the conversion of raw `.edf` and `.csv` files into preprocessed `.fif` continuous data files, which will be ready for the sequence prediction model.

### **Execution**
```bash
uv run custom/preprocess_rawEEG.py
```

### **Data Structure & Shapes**
- **Input shape/format:** 
  - `edf_signals.edf`: Raw continuous EEG/PSG signals of variable duration and arbitrary multi-channel layout.
  - `csv_hypnogram.csv`: Tabular sleep stage annotations containing timing. Evaluated as continuous epochs.
- **Output shape/format:** 
  - `[subject_id]_preprocessed.fif`: MNE Raw FIF format. Continuous signal containing only standardized EEG derivations and properly attached `mne.Annotations`.

### **What happens in the code (Step-by-Step)**
1. **Parses Hypnogram:** Maps sleep stages to standard labels and checks the fraction of Non-Scored (NS) epochs.
2. **Loads EDF Header:** Inspects the raw EDF without fully loading it into RAM to apply initial guardrails (length check).
3. **Channel Selection:** Selects only EEG channels using `raw.pick_types(eeg=True, exclude="bads")`.
4. **Filtering:** Applies a Band-pass filter (0.5–40 Hz) and a Notch filter (50/60 Hz).
5. **Sampling Rate Conversion:** Resamples the continuous data to **256 Hz** if it is not already.
6. **Bad Channel Interpolation:** Detects bad channels based on standard deviation boundaries and interpolates them.
7. **Annotations:** Attaches the parsed 30-second epoch sleep stages to the signal structure.
8. **Export:** Saves the preprocessed continuous raw data as a `.fif` file into the `preprocessing_output` directory.

### **Rules & Guardrails defined in Code**
- **`MIN_DURATION_HOURS = 1.5`**: Skips the subject entirely if the total recording duration is less than 1.5 hours (5400 seconds).
- **`NS_SKIP_THRESHOLD = 0.50`**: Skips the subject if the fraction of Non-Scored (NS) epochs in the hypnogram represents more than 50% of the recording.
- **`TARGET_SFREQ = 256`**: Target sampling rate is strictly enforced to 256 Hz.
- **`EPOCH_DURATION_S = 30.0`**: Epochs are strictly processed as 30-second windows.
- **`BAD_CH_THRESHOLD = 5`**: An EEG channel is considered "bad" and marked for interpolation if its standard deviation is greater than `5 * median` or less than `median / 5`. 

---

## 2. Sequence Prediction (`sequence_prediction.py`)

This script evaluates the preprocessed `.fif` signals directly using a pre-trained BendingCollegeWav2Vec (BENDR) pipeline.

### **Execution**
```bash
python custom/sequence_prediction.py
```

### **Data Structure & Shapes**
- **Input:** 
  - `[subject_id]_preprocessed.fif` (Loaded from the internal `preprocessing_output` directory).
- **Data Shape transformations:**
  1. **Raw to Tensor:** Extracts specific EEG channels and converts the data block to a numpy array, then to a PyTorch Tensor of shape `(1, channels, total_samples)`.
  2. **Sequence Length Cropping/Padding:** Extends or truncates the time dimension (`total_samples`) to exactly **15360 samples** (which is exactly 60 seconds of data at 256 Hz). If shorter, zero-padding is appended.
  3. **Channel Formatting:** The internal BENDR architecture expects 20 channels to safely load the pre-trained weights `ConvEncoderBENDR(20, ...)`. If the dataset has fewer than 20 channels, dummy/zero channels are concatenated up to exactly 20. 
  - **Final Tensor Shape fed to model:** `(1, 20, 15360)` representing `(Batch, Channels, Sequence_Length)`
- **Output:** 
  - `seq_results.csv` and `seq_results.xlsx`: Summarized metrics including Loss, Accuracy, and Mask % for each evaluated subject.
  - `best_model.pt`: The model checkpoint holding the weights that produced the highest accuracy during evaluation.
  - `training_log.csv`: Sequential raw execution log containing timing and sample processing speed.

### **What happens in the code (Step-by-Step)**
1. **Model Initialization:** Instantiates `ConvEncoderBENDR` and `BENDRContextualizer` and loads pre-trained weights (`encoder.pt` and `contextualizer.pt`).
2. **Data Loading:** Iterates over all `.fif` files previously built.
3. **Channel Filtering:** Explicitly keeps only 7 specific EEG derivations (`EEG F3-A2, EEG F4-A1, EEG A1-A2, EEG C3-A2, EEG C4-A1, EEG O1-A2, EEG O2-A1`). Missing channels are warned and later zero-padded.
4. **Resampling Verification:** Ensures loaded data is mapped to `256 Hz`.
5. **Tensor Conversion:** Transforms the continuous array block into a fixed-size `(1, 20, 15360)` tensor footprint.
6. **Forward Pass:** Feeds the formatted tensor to the `BendingCollegeWav2Vec` model in evaluation mode (`process.train(False)`) to extract representations and calculate the masking objective loss + sequence accuracy.
7. **Logging & Checkpointing:** Saves iterative results to CSVs and exports the best performing state dictionary.

### **Hyperparameters & Rules defined in Code**
- **`FORCE_CPU = True`**: Hardcoded hardware fallback constraint. Forces the network to compute on the CPU to prevent common cuDNN configuration issues. You can toggle this to `False` if your GPU/PyTorch stack is fully compatible.
- **`SAMPLING_RATE = 256`**: The BENDR standard sampling rate.
- **`SEQUENCE_LENGTH = 15360`**: Evaluates standardized 60-second contiguous windows.
- **`HIDDEN_SIZE = 512`** and **`LAYER_DROP = 0.01`**: Fixed architecture size definitions.
- **`EEG_CHANNELS` Constraint list**: Explicit script rule to extract only listed Polysomnography derivations, aggressively discarding EOG, EMG, and auxiliary channels inside the parent `.fif` file before casting to a PyTorch tensor.

---

## Workspace Directory Management

### 1. Data Input (`RAW`)
The raw dataset directory must be kept two levels outside the script location (`../../rawEEG`).
```text
rawEEG/
└── [subject_id]/
    ├── edf_signals.edf       (Required: Multi-channel continuous recording)
    ├── csv_hypnogram.csv     (Required: Epoch timing mapping)
    └── csv_events.csv        (Optional)
```

### 2. Pretrained Weights
Model weights must be present inside the root `weights/` directory for `sequence_prediction.py` to correctly load states.
```text
BENDR/weights/
├── encoder.pt            (Pretrained weights for ConvEncoderBENDR)
└── contextualizer.pt     (Pretrained weights for BENDRContextualizer)
```

### 3. Pipeline Output Generation
Outputs are procedurally generated in the local folders after script executions.
```text
custom/
├── preprocessing_output/
│   └── [subject_id]_preprocessed.fif  (Step 1: Generated files)
│
└── results/
    ├── seq_results.csv                (Step 2: Table Summary)
    ├── seq_results.xlsx               (Step 2: Excel Reports)
    ├── training_log.csv               (Step 2: Training Log)
    └── best_model.pt                  (Step 2: Network Checkpoint)
```
