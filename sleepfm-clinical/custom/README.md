# SleepFM Clinical Custom Pipeline

This repository contains an end-to-end clinical pipeline tailored for analyzing Polysomnography (PSG) and raw EEG data utilizing the **SleepFM** framework. 
The pipeline seamlessly transitions across four phases: **EDF to HDF5 Conversion → Embedding Generation → Sleep Staging Inference → Disease (Diagnosis) Prediction**.

Importantly, this script is explicitly designed for ultra-low RAM usage. It loads a single subject into memory at a time and aggressively deletes intermediate data files and tensors (operating `gc.collect()` and `cuda.empty_cache()`) per pipeline step.

---

## 1. Execution Order

There is no separate preprocessing script needed. Everything runs sequentially within the monolithic `main.py` entry point. Configuration variables (Paths, Batches) are hardcoded directly into the `main()` function.

```bash
python custom/main.py
```

---

## 2. Input Data Structure

The pipeline is intelligent enough to automatically scan for two different recording layout typologies inside the base `DATA_DIR`:

**Layout A (Flat layout):**
```text
rawEEG/
├── 00000001.edf              (Raw continuous signals)
├── 00000001.csv              (Sleep stage hypnogram tabular data)
├── demo_age_gender.csv       (Demographics matrix: Required for Phase 4)
├── is_event.csv              (Labels matrix: Required for Phase 4)
└── time_to_event.csv         (Timing matrix: Required for Phase 4)
```

**Layout B (Sub-folder layout):**
```text
rawEEG/
├── 00000001/
│   ├── edf_signals.edf       (Raw continuous signals)
│   └── csv_hypnogram.csv     (Sleep stage hypnogram tabular data)
├── demo_age_gender.csv       
├── is_event.csv              
└── time_to_event.csv         
```

---

## 3. What happens in the code (Step-by-Step)

The `custom/main.py` orchestrates the flow iteratively for each subject discovered in the input directory.

1. **System / CUDA Initialization:** Preloads NVIDIA `cuDNN` library modules dynamically via `ctypes` and disables PyTorch NVFuser (`PYTORCH_NVFUSER_DISABLE="1"`) to prevent hardware freezing.
2. **Phase 1: Hypnogram Validation:** Parses the clinical sleep stages. Discards any regions labeled `NS` (Not Scored) or `Unknown`. Validates total acceptable lengths and re-formats structural time blocks from standard 30-second windows into **5-second sub-epochs** for embedding continuity.
3. **Phase 2: EDF to HDF5 Preprocessing:** Processes the continuous `.edf` dataset using `EDFToHDF5Converter()`. This applies the standard resampling algorithms and produces an intermediate `.hdf5` state. It performs sanity checks on shapes logging `NaNs`, `Infs`, and High-Zero percentages per channel.
4. **Phase 3: Deep Embedding Generation:** Generates compressed representational structures utilizing a `SetTransformer` pretrained model. It outputs two data scales:
   - **5-second granular blocks:** Saved to `embeddings/` directly mirroring original sub-epochs.
   - **5-minute aggregated blocks:** Saved to `embeddings_5min/`.
   - *Memory Purge:* The intermediate large raw `.hdf5` file is destroyed to free disk and RAM resources here.
5. **Phase 4: Sleep Staging:** Pulls the granular `embeddings/` mapping them against the processed Hypnograms using a finetuned Staging model. Logs a `.csv` tracking actual target targets against predicted stage properties (Wake, N1, N2, N3, REM) probabilities.
6. **Phase 5: Disease Diagnosis:** Fuses extracted SetTransformer embeddings alongside `demo_age_gender.csv` table features to generate survival models/hazard scores per patient via Finetuned COXPH LSTM networks. Checks against `is_event.csv`.
7. **Cleanup:** Destroys batch residuals, builds the terminal iteration stats per patient, tracking successes in `summary.csv`.

---

## 4. Hyperparameters, Shapes & Rules defined in Code

This specific script implements rigid technical guidelines optimized for the underlying PyTorch architectures:

### Signal Constraints
* **`TARGET_SFREQ = 128`**: The internal standard **Resample Rate** for this framework is fixed to **128 Hz**. All continuous streams are inherently downgraded or evaluated to meet this scale inside the `EDFToHDF5` module.
* **Epoch Length Rules**: The pipeline structurally relies on **5-second epochs**. Standard actual clinical sleep forms report on 30-second logic. The script (`validate_hypnogram()` function) calculates `epoch_sec // 5` systematically fracturing 1 clinical epoch into 6 parallel sub-epochs containing `EmbeddingNumber` index targets.

### Guardrails
* **`MIN_MINUTES = 15.0`**: Will aggressively drop / skip any recording layout featuring less than 15 valid minutes of clean staged data.
* **`NS_THRESHOLD = 0.50`**: Evaluates total dataset integrity. Any file with more than 50% non-scored or completely unknown variables gets eliminated before conversion takes place.
* **Architecture Flexibility:** Unlike standard CNN benchmarks counting exact tensor matches, the PyTorch `SetTransformer` used in this architecture evaluates variable multi-channel mappings natively across blocks without catastrophic failures due to `N/A` channels.

### Structural Shapes in Code
1. **SleepStaging Dataclass (`_SleepStagingDataset`)** returns packed sequences constrained via `max_seq_len` padding logic:
   - Evaluated Model Data Shape: Multi-dimensional chunks formatting `padded_x` containing `(Channels, seq_len, Embeddings)` mixed alongside masking arrays.
2. **Diagnosis Dataclass (`_DiagnosisDataset`)** returns unified sequences bound to Demographic parameters:
   - Base features tensor evaluated alongside explicit physical factors represented as single arrays: `event_time`, `is_event`, `demo_feats`.

---

## 5. Pipeline Output Generation (Results)

The `OUTPUT_DIR` evaluates sequentially and produces structural logs across:
```text
custom/results/
├── pipeline.log                  (Detailed terminal system execution trace)
├── summary.csv                   (Summary report listing Pass/Fail metrics for each subject)
│
├── embeddings/
│   └── [StudyID].hdf5            (Generated Granular 5-sec Tensor Blocks)
├── embeddings_5min/
│   └── [StudyID].hdf5            (Generated Aggregated 5-min Tensor Blocks)
│
├── sleep_staging/
│   └── [StudyID].csv             (Targets versus Sequence Probabilities Table [Wake, N1, N2...])
└── diagnosis/
    └── [StudyID].csv             (Predicted Hazard Score matrix)

```
