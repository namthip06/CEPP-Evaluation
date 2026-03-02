# BENDR Custom Pipeline: Sequence Prediction Report

## 1. Overview of the Evaluation
This report summarizes the performance of the **BENDR** (BendingCollegeWav2Vec) pre-trained model on custom Polysomnography (PSG) datasets. The evaluation was conducted using a standardized sequence length window of **15,360 samples** (representing 60 seconds at 256 Hz) with a masking percentage (`mask_pct`) set to **30% (0.3)**.

The results are generated and logged in `seq_results.csv` and `training_log.csv`.

## 2. Summary of Results
A total of **41 subjects** were evaluated in this pipeline. 
The key performance metrics extracted from the log files are as follows:

* **Accuracy Range:** **2.5%** to **44.37%** (0.025 to 0.4437)
* **Best Performing Subject:** `00000449-159547` achieved the highest accuracy of **44.37%** with the lowest recorded Loss of **3.4667**.
* **Loss Range:** **3.46** to **4.57**
* **Processing Speed:** Averaged between **2,055** and **2,311** samples per second, taking approximately **6.6** to **7.4** seconds to evaluate a single 60-second window block on the processing hardware (CPU).

*Note: The overall accuracy numbers are significantly lower than optimal because of the channel sparsity limitations detailed below.*

---

## 3. The 10-20 Standard System vs. Real World Data

### The Standard 20-Channel Assumption (BENDR expectation)
In EEG research, especially when utilizing large-scale pre-trained models like BENDR, architectures are heavily reliant on the **10-20 System**. A standard 20-channel arrangement covers the primary cortical regions of the brain. The pretrained model weights (`encoder.pt` and `contextualizer.pt`) were originally trained to extract complex **spatial correlations** across these standard electrodes:

* **Frontal (Front):** `Fp1, Fp2, F7, F3, Fz, F4, F8`
* **Central (Middle):** `T3 (or T7), C3, Cz, C4, T4 (or T8)`
* **Parietal (Back/Side):** `P3, Pz, P4`
* **Temporal (Side):** `T5 (or P7), T6 (or P8)`
* **Occipital (Back):** `O1, O2`
* **Reference/Ground:** `A1, A2 (Ears) or Ref`

### The Real Data Scenario (Polysomnography)
Our real dataset consists of **34 total channels** because it is a full Polysomnography (PSG) sleep study, not purely an EEG recording. Most of these channels represent physiological metrics, not brainwaves (e.g., `EMG Chin`, `ECG I/II`, `SpO2`, `RespRate`, `Pleth`). 

The pipeline strictly filters the available channels looking for usable EEG derivations. However, the real dataset contains **only 7 valid EEG derivations**:
1. `EEG F3-A2`
2. `EEG F4-A1`
3. `EEG A1-A2`
4. `EEG C3-A2`
5. `EEG C4-A1`
6. `EEG O1-A2`
7. `EEG O2-A1`

The remaining 27 physical channels are discarded because the BENDR network is structurally incompatible with them.

---

## 4. Architectural Workaround: Zero-Padding

### The Problem
The primary problem we face is a shape mismatch. The `ConvEncoderBENDR` architecture was hard-coded and pretrained to accept exactly **20 input channels**. 
Because the model relies on standard 20-channel pretrained weights, **we cannot retroactively manipulate the internal `n_channels` variable of the network down to 7**. Doing so would destroy the structural integrity of the pretrained weights and break the forward pass.

### The Code Implementation (Zero Padding)
To ensure the hardware pipeline does not crash, the code enforces a geometric workaround known as **Zero-Padding**.

1. The code extracts the 7 continuous EEG arrays from the preprocessed `.fif` file.
2. The code groups them into a tensor: `Shape: (1, 7, 15360)`.
3. To reach the required minimum, it automatically generates 13 dummy channels entirely populated by `0.0`.
4. It concatenates the real 7 channels with the 13 dummy channels.
5. The final tensor shape fed into the network becomes: `(1, 20, 15360)` or `(Batch, Channels, Sequence_Length)`.

### Technical Implications
While zero-padding allows the code to execute successfully and the shapes to resolve, **it severely penalizes the predictive accuracy** of the network:
- The BENDR model learns through spatial topography (how electrical waves travel from one point of the head to another). 
- Because we only provide 7 out of 20 active spatial nodes, the model is "blind" to more than half of the expected brain map. 
- Evaluating data with 13 blank (zeroed) channels prevents the model from cross-referencing activity in the missing parietal, central, and temporal areas, which directly contributes to the lower range of accuracy observed in the logs (averaging ~12-15%).
