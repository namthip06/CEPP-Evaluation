# L-SeqSleepNet Data Preparation (Custom)

This folder contains customized Python scripts for preprocessing and preparing raw EEG/EOG sleep data so that it can be used for training, testing, or evaluation with the L-SeqSleepNet model. 

The pipeline takes raw signals and clinical annotations (.edf and .csv files), extracts and filters relevant features, computes spectrograms, and outputs structured `.mat` files and text file lists as required by the model.

---

## Folder Structure

```text
custom/
├── prepare_raw_data.py    # Step 1: Raw data extraction & preprocessing
├── prepare_data.py        # Step 2: Spectrogram generation
├── gen_file_list.py       # Step 3: File list generation for model input
├── processed_data/        # Directory containing intermediate structured data
├── mat/                   # Directory containing final dual-stream model data
└── file_list/             # Directory containing paths to data files
    ├── eeg/               # EEG file lists
    └── eog/               # EOG file lists
```

---

## File Types & Descriptions

### 1. Python Scripts (`.py`)
These are the core execution scripts that constitute the data processing pipeline.

- **`prepare_raw_data.py`**
  - **What it does:** Reads raw `.edf` signal files and `.csv` hypnogram/event annotations from the `../../rawEEG` directory. It selects specific EEG and EOG channels, applies bandpass filtering (0.3 - 40.0 Hz), resamples data to 100 Hz, chunks the signals into 30-second epochs based on the hypnogram, and aligns them with normalized sleep stage labels (W, N1, N2, N3, R). It removes unscorable (NS) epochs.
  - **Output:** Saves the extracted raw signal segments (`data`), numeric labels (`label`), and one-hot encoded labels (`y`) as `.mat` files in the `processed_data/` folder.

- **`prepare_data.py`**
  - **What it does:** Reads the intermediate `.mat` files from `processed_data/`. It processes each channel (EEG and EOG) independently to compute Short-Time Fourier Transform (STFT) spectrograms using a 2-second Hamming window with a 1-second overlap. It calculates the log magnitude spectrum (`20*log10`).
  - **Output:** Generates dual-stream `.mat` files (with naming conventions like `n01_1_eeg.mat`) saved into the `mat/` directory. Each file contains `X1` (time-domain raw signal) and `X2` (frequency-domain spectrogram), along with `label` and `y`.

- **`gen_file_list.py`**
  - **What it does:** Scans the `mat/` folder and creates text files that L-SeqSleepNet will use to locate the data. It counts the number of samples (epochs) directly from the `.mat` labels and formats them.
  - **Output:** Generates `test_list.txt` under `file_list/eeg/` and `file_list/eog/`. Each file contains lines detailing the relative path to the `.mat` file and its total number of samples separated by a tab.

### 2. Data Directories (`.mat` & `.txt`)

- **`processed_data/`**
  - **File Type:** `.mat` (MATLAB Data Dictionary)
  - **Content:** Intermediate storage. Contains raw time-series matrices of size `(N, 3000, 2)` where `N` is the number of valid 30s epochs, `3000` samples per epoch (at 100Hz), and `2` channels (EEG, EOG).
  
- **`mat/`**
  - **File Type:** `.mat` (MATLAB Data Dictionary)
  - **Content:** Final model-ready data separated by channel. Contains `X1` (Raw signals, `N x 3000`), `X2` (Spectrogram, `N x 29 x 129`), `label` (integer labels), and `y` (one-hot labels).

- **`file_list/`**
  - **File Type:** `.txt` (Plain text)
  - **Content:** Mapping files used by the dataset loader during training/inference. Contains relative paths to files in the `mat/` folder and their respective epoch counts.

---

## Steps to Run the Pipeline

To fully prepare the dataset from raw EDF files to L-SeqSleepNet compatible formats, execute the scripts in the following sequential order:

### **Step 1: Extract and Preprocess Raw Data**
Extracts waveforms, applies filters, matches with annotations, and chunks into 30-second epochs.
*(Requires your raw data to be located in `../../rawEEG/[subject_id]/`)*
```bash
python prepare_raw_data.py
```
*Check output in the `processed_data/` directory.*

### **Step 2: Generate Dual-Stream Spectrograms**
Transforms the time-domain data into spectrograms required for the model's feature extraction.
```bash
python prepare_data.py
```
*Check output in the `mat/` directory.*

### **Step 3: Generate File Lists**
Creates the `.txt` reference files that the model dataloader expects.
```bash
python gen_file_list.py
```
*Check output in the `file_list/` directory (`file_list/eeg/test_list.txt` and `file_list/eog/test_list.txt`).*

After completing these steps, the setup in this directory is fully prepared, and you are ready to use the `mat/` files and `file_list/` files for L-SeqSleepNet model training or evaluation.
