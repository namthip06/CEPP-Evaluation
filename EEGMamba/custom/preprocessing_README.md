## 📊 Data Preprocessing for Evaluation

### Do I Need to Preprocess My Data?

**Yes, preprocessing is required** for both training and evaluation data. The EEGMamba model expects preprocessed EEG data in a specific format. Raw EEG signals must undergo several preprocessing steps before they can be used for evaluation.

### Why Preprocessing is Necessary

The model was trained on preprocessed data with specific characteristics:
- **Sampling rate**: 200 Hz
- **Frequency filtering**: Bandpass filtered (0.3-35 Hz or 0.3-52 Hz depending on application)
- **Notch filtering**: Power line noise removal (50/60 Hz)
- **Normalization**: Data scaled appropriately (typically divided by 100)
- **Segmentation**: Data organized into fixed-length epochs

Using raw, unprocessed data will result in poor model performance or errors.

### Preprocessing Pipeline

#### For Sleep Staging (e.g., ISRUC Dataset)

Based on the preprocessing scripts in [`preprocessing/preprocessing_for_finetuning/ISRUC/`](preprocessing/preprocessing_for_finetuning/ISRUC/), here are the required steps:

1. **Load Raw EEG Data**
   ```python
   import mne
   from mne.io import read_raw_edf
   
   raw = read_raw_edf('path/to/your/file.edf', preload=True)
   ```

2. **Select Channels**
   ```python
   # Select the specific channels used in your study
   # Example for ISRUC: 6 channels (indices 2-8)
   channels = ['C3-A2', 'C4-A1', 'F3-A2', 'F4-A1', 'O1-A2', 'O2-A1']
   raw.pick_channels(channels)
   ```

3. **Apply Filtering**
   ```python
   # Bandpass filter: 0.3-35 Hz
   raw.filter(0.3, 35, fir_design='firwin')
   
   # Notch filter: Remove 50 Hz power line noise (use 60 Hz for US data)
   raw.notch_filter(50)
   ```

4. **Resample to 200 Hz** (if needed)
   ```python
   raw.resample(sfreq=200)
   ```

5. **Extract and Segment Data**
   ```python
   import numpy as np
   
   # Convert to numpy array
   psg_array = raw.get_data().T  # Shape: (n_samples, n_channels)
   
   # For sleep staging: segment into 30-second epochs at 200 Hz
   epoch_length = 30 * 200  # 6000 samples per epoch
   
   # Remove incomplete epochs
   remainder = psg_array.shape[0] % epoch_length
   if remainder > 0:
       psg_array = psg_array[:-remainder, :]
   
   # Reshape into epochs
   psg_array = psg_array.reshape(-1, epoch_length, n_channels)
   
   # Group into sequences (e.g., 20 epochs per sequence)
   seq_length = 20
   remainder = psg_array.shape[0] % seq_length
   if remainder > 0:
       psg_array = psg_array[:-remainder, :, :]
   
   psg_array = psg_array.reshape(-1, seq_length, epoch_length, n_channels)
   # Transpose to (n_sequences, seq_length, n_channels, epoch_length)
   epochs_seq = psg_array.transpose(0, 1, 3, 2)
   ```

6. **Save Preprocessed Data**
   ```python
   # Save each sequence
   for i, seq in enumerate(epochs_seq):
       np.save(f'preprocessed_data/seq_{i}.npy', seq)
   ```

#### For General EEG Data (Pretraining Format)

Based on [`preprocessing/preprocessing_for_pretraining/preprocessing_raw_for_pretraining.py`](preprocessing/preprocessing_for_pretraining/preprocessing_raw_for_pretraining.py):

1. **Select Standard Channels**
   ```python
   channels = [
       'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
       'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5',
       'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
       'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8',
       'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8',
       'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
   ]
   raw.pick_channels(channels, ordered=True)
   ```

2. **Set Reference and Filter**
   ```python
   # Average reference
   raw.set_eeg_reference(ref_channels='average')
   
   # Resample to 200 Hz
   raw.resample(200)
   
   # Bandpass filter: 0.3-52 Hz
   raw.filter(l_freq=0.3, h_freq=52)
   
   # Notch filter: 60 Hz (adjust based on your region)
   raw.notch_filter(60)
   ```

3. **Segment into 5-second epochs**
   ```python
   samples = raw.get_data(units='uV')
   
   # Remove incomplete samples
   temp = samples.shape[1] % 1000
   if temp != 0:
       samples = samples[:, :-temp]
   
   # Reshape: (n_channels, n_samples) -> (n_channels, n_epochs, 5, 200)
   samples = samples.reshape(64, -1, 5, 200)
   
   # Transpose to (n_epochs, n_channels, 5, 200)
   samples = samples.transpose(1, 0, 2, 3)
   ```

### Data Format Expected by Model

The model expects data in the following format:

**Input Shape**: `(batch_size, num_channels, time_segments, points_per_patch)`

For example:
- **Sleep staging**: `(batch_size, 6, 20, 6000)` or reshaped appropriately
- **General EEG**: `(batch_size, 64, 5, 200)` or `(batch_size, 22, 4, 200)` for specific applications

**Normalization**: Data is typically divided by 100 (as seen in [`datasets/isruc_dataset.py`](datasets/isruc_dataset.py) line 28: `return seq/100, label`)

### Quick Preprocessing Example

Here's a complete example for preprocessing a single EEG file:

```python
import mne
import numpy as np

# Load raw data
raw = mne.io.read_raw_edf('your_data.edf', preload=True)

# Select channels (adjust based on your data)
channels = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']
raw.pick_channels(channels)

# Preprocessing
raw.set_eeg_reference(ref_channels='average')
raw.resample(200)
raw.filter(0.3, 35, fir_design='firwin')
raw.notch_filter(50)  # or 60 Hz

# Get data and normalize
data = raw.get_data(units='uV')
data = data / 100  # Normalization

# Segment (example: 30-second epochs)
epoch_samples = 30 * 200
remainder = data.shape[1] % epoch_samples
if remainder > 0:
    data = data[:, :-remainder]

data = data.reshape(len(channels), -1, epoch_samples)
data = data.transpose(1, 0, 2)  # (n_epochs, n_channels, samples_per_epoch)

# Save
np.save('preprocessed_data.npy', data)
```

### Important Notes

> [!IMPORTANT]
> - **Always preprocess evaluation data** using the same pipeline as training data
> - **Match the sampling rate** (200 Hz is standard for EEGMamba)
> - **Use consistent filtering** parameters across all data
> - **Normalize appropriately** (typically divide by 100)

> [!WARNING]
> - Different datasets may require different channel selections
> - Epoch lengths vary by application (30s for sleep staging, 5s for general EEG)
> - Power line frequency (50 Hz vs 60 Hz) depends on your geographic region

> [!TIP]
> - Check the preprocessing scripts in [`preprocessing/`](preprocessing/) for dataset-specific examples
> - Verify your preprocessed data shape matches the model's expected input
> - Use MNE-Python for robust EEG preprocessing