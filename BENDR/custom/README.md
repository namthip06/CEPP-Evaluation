# Technical Blockers in BENDR Sleep Staging Evaluation

This document outlines the critical issues encountered during the execution of the sleep staging evaluation script. These blockers prevented the model from completing the inference process.

## 1. Architectural Inconsistency (Weight Mismatch)

The pre-trained BENDR weights were found to be incompatible with the current dataset's input configuration.

* **Cause:** The pre-trained `ConvEncoderBENDR` was trained on **20-channel** EEG data (standard 10-20 system). However, the evaluation script attempted to initialize a model with a different channel count (e.g., 2 channels from Sleep-EDF).
* **Result:** A `size mismatch` error occurred in the first layer (`Encoder_0.0.weight`), as the weight tensors in the checkpoint did not match the shape of the tensors in the instantiated model.

## 2. Data Type & Structure Conflict (AttributeError)

A failure occurred during the transformation of preprocessed EEG data into a format suitable for the neural network.

* **Cause:** The script expected a PyTorch `Tensor` object to perform NumPy conversion via `.numpy()`. Instead, it received a Python `list` object.
* **Result:** The execution halted with an `AttributeError: 'list' object has no attribute 'numpy'`, indicating a failure in the data pipeline's handling of transformed epoch batches.

## 3. Metadata Discrepancy (Annotation Missing)

The script failed to synchronize the raw EEG signals with their corresponding sleep stage labels.

* **Cause:** The script utilized `mne.events_from_annotations` with a hardcoded mapping dictionary (`STAGE_TO_INT`). The string descriptions for sleep stages in the `.fif` files (e.g., 'W', 'Stage 1', etc.) did not match the expected keys in the dictionary.
* **Result:** A `ValueError: Could not find any of the events you specified` was triggered, preventing the extraction of labeled epochs for validation.

## 4. Environment & Legacy Code Conflicts

The DN3 library (v0.2-alpha) used in this project relies on legacy Python patterns that are incompatible with modern Python versions (3.10+).

* **Cause:** * **Module Naming:** Changes in the `pyyaml-include` library (v2.x) renamed the package, breaking the old `import yamlinclude` statement.
* **Standard Library Deprecations:** Python 3.10 moved `Iterable` from `collections` to `collections.abc`, which caused immediate import failures within the core DN3 utility files.


* **Result:** Multiple `ModuleNotFoundError` and `ImportError` exceptions prevented the library from initializing correctly.