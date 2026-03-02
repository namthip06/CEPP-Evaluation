# Error Report: L-SeqSleepNet Evaluation Pipeline

This document outlines the detailed causes of the errors encountered when trying to run the inference script (`test_lseqsleepnet.py`) and standard `evaluate.py` via the custom data preparation pipeline, alongside the steps required to fix them.

## 1. Checkpoint Graph Shape Mismatch `(InvalidArgumentError)`
**Error snippet:**
```
tensorflow.python.framework.errors_impl.InvalidArgumentError: Restoring from checkpoint failed. 
Assign requires shapes of both tensors to match. lhs shape= [10,256] rhs shape= [20,256]
```
**Why it failed:** 
The script `test_lseqsleepnet.py` has a default configuration parameter of `--nsubseq 10` (number of subsequences = 10). However, the pre-trained model you are trying to load (`__pretrained_shhs_1chan_subseqlen10_nsubseq20_1blocks`) expects a tensor generated from `--nsubseq 20`. Because the number of sequence parts fundamentally changes the tensor size expected in the RNN and attention mechanisms, TensorFlow crashed when trying to populate a `[10,...]` sized layer with `[20,...]` sized weights from the checkpoint. 

**Solution:**
You must explicitly override the argument when calling the script to match the pretrained model's configuration:
`--nsubseq 20`

## 2. Invalid HD5F `.mat` Format `(OSError / FileNotFoundError)`
**Error snippet:**
```
File "h5py/h5f.pyx", line 88, in h5py.h5f.open
OSError: Unable to open file (file signature not found)
```
**Why it failed:** 
The original model code in `datagenerator_from_list_v3.py` attempts to use `h5py` to read the `.mat` data files. `h5py` only strictly supports MATLAB v7.3 `HDF5` configurations. When the `prepare_data.py` code generated your files in the `mat/` folder, it generated them in a lower version that `h5py` does not recognize as valid HDF5.
**Solution:**
The file reader inside `datagenerator_from_list_v3.py` and `datagenerator_wrapper.py` needs to be swapped to use `hdf5storage.loadmat(filename)` or `scipy.io.loadmat(filename)` instead of `h5py`, which cleanly parses both formats.

## 3. Path Resolution Failures 
**Error snippet:**
```
FileNotFoundError: [Errno 2] No such file or directory: '../../mat/n01_1_eeg.mat'
```
**Why it failed:** 
The file lists (`test_list.txt`) store relative paths (e.g., `../../mat/n01.mat`). When you execute `test_lseqsleepnet.py` from the `/custom` directory, the relative path points sequentially to the wrong directory because the script does not resolve the relative path *based on the location of the `.txt` file*, it resolves it based on your current terminal Working Directory.
**Solution:**
Inject an absolute resolution handler in the loader (`datagenerator_wrapper.py`), converting paths to `os.path.normpath(os.path.join(filelist_dir, items[0]))` so they correctly link back up to the `mat/` folder.

## 4. Inconsistent Formatting `(TabError)`
**Error snippet:**
```
TabError: inconsistent use of tabs and spaces in indentation
```
**Why it failed:** 
Python 3 strictly disallows mixing tabs and spaces. During some of our code edits, `datagenerator_wrapper.py` was introduced to an incompatible combination of whitespace. 
**Solution:**
Formatted the entire file to use unified 4-space tabulations. 

## 5. Missing Dependencies
**Why it failed:**
The environment lacked several libraries. Most critically, because the model code was written circa 2019, `tensorflow==2.0+` deprecates `tensorflow.contrib`, meaning the RNN configurations fail immediately. 
**Solution:**
The environment strictly requires `tensorflow==1.15.5` to execute properly alongside dependencies such as `hdf5storage`, `scikit-learn`, and `numpy<2`.

---
### Current State
These bugs have now been securely patched in the codebase. You can trigger the evaluation properly initialized with the correct shapes using:
```bash
uv run ../sleepedf-20/network/lseqsleepnet/test_lseqsleepnet.py \
    --eeg_train_data file_list/eeg/test_list.txt \
    --eeg_test_data file_list/eeg/test_list.txt \
    --out_dir ./output \
    --checkpoint_dir ../sleepedf-20/__pretrained_shhs_1chan_subseqlen10_nsubseq20_1blocks \
    --nsubseq 20
```
*(Note: As seen in the final execution logs, evaluating on the entire test list simultaneously processes gigantic tensors, causing memory allocations into the Gigabytes range; expect runtime duration accordingly).*
