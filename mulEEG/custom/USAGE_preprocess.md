# preprocess.py Usage Guide

## Purpose
The `preprocess.py` script is designed to batch preprocess raw EEG data (EDF format) and corresponding hypnograms (CSV format) into a format compatible with the mulEEG pipeline (`.npz` files).

It performs the following operations:
1.  Reads raw EDF signals.
2.  Parses CSV hypnograms and maps sleep stages to numeric labels (0=Wake, 1=N1, 2=N2, 3=N3/N4, 4=REM).
3.  Selects a specific EEG channel (or auto-selects the first available EEG channel).
4.  Resamples the signal to 100 Hz if necessary.
5.  Aligns the signal length with the hypnogram labels.
6.  Splits the data into 30-second epochs.
7.  Optionally trims excessive wake periods at the beginning and end of the recording.
8.  Saves the processed data as `.npz` files.

## Prerequisites & Folder Structure
The script expects a specific folder structure relative to its location (`mulEEG/custom/`).

Input data should be located in a `rawEEG` directory at the same level as the `mulEEG` project directory:

```text
/Documents/CEPP/
├── mulEEG/
│   └── custom/
│       └── preprocess.py
└── rawEEG/
    ├── [patient_id_1]/
    │   ├── edf_signals.edf      <-- Required
    │   └── csv_hypnogram.csv    <-- Required
    └── [patient_id_2]/
        ...
```

## Configuration
Configuration variables are located at the beginning of the `main()` function in `preprocess.py`. You can modify these values directly in the script to adjust behavior.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `select_channel` | `None` | The specific EEG channel to extract (e.g., `'EEG C4-A1'`). If `None`, it automatically selects the first channel containing "EEG". |
| `trim_wake_edges` | `True` | If `True`, removes excessive wake periods from the start and end of the recording to focus on sleep. |
| `edge_minutes` | `30` | The amount of wake time (in minutes) to keep before the first sleep stage and after the last sleep stage when trimming. |
| `epoch_sec_size` | `30` | The duration of each epoch in seconds. |

## Usage
To run the preprocessing script, execute it from the terminal using Python:

```bash
# Navigate to the custom directory
cd /home/nummm/Documents/CEPP/mulEEG/custom

# Run the script
python preprocess.py
```

## Output
Processed files are saved to the `preprocessing_output` directory within `mulEEG/custom/`.

```text
mulEEG/custom/preprocessing_output/
├── [patient_id_1]/
│   └── [patient_id_1].npz
└── [patient_id_2]/
    └── [patient_id_2].npz
```

Each `.npz` file contains:
*   `x`: EEG signal epochs (Shape: `[n_epochs, 3000]`)
*   `y`: Sleep stage labels (Shape: `[n_epochs]`)
*   `fs`: Sampling rate (100 Hz)
