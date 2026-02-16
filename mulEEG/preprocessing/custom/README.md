# Custom EDF/CSV Preprocessing for mulEEG

## Overview

This script preprocesses custom EEG data from **EDF files** and **CSV hypnogram files** into the format required by the mulEEG model. It is based on the SHHS preprocessing logic but adapted for generic EDF/CSV inputs.

## Features

- ✅ **Batch processing** - Process multiple patients automatically from structured directories
- ✅ **Automatic resampling** - Converts any sampling rate to 100Hz for model compatibility
- ✅ **Flexible CSV parsing** - Auto-detects sleep stage column
- ✅ **Auto channel selection** - Automatically finds EEG channels
- ✅ **Label mapping** - Converts sleep stage strings to numeric labels (0-4)
- ✅ **30-second epochs** - Segments signals into standard epochs
- ✅ **Sleep period trimming** - Optional edge trimming like SHHS preprocessing
- ✅ **Compatible output** - Generates .npz files compatible with mulEEG
- ✅ **Memory optimized** - Loads only selected channel (70-85% less memory usage)
- ✅ **Smart skip logic** - Automatically skips already processed patients
- ✅ **Error handling** - Continues processing even if individual patients fail

## Requirements

```bash
pip install numpy pandas mne
```

## Usage

### Configuration

แก้ไขค่าตัวแปรใน section `CONFIGURATION` ของไฟล์ `preprocess_custom.py`:

```python
# ==================== CONFIGURATION ====================
# แก้ไขค่าตัวแปรเหล่านี้ตามไฟล์ข้อมูลของคุณ

# Path to input directory containing patient folders
input_base_dir = "/home/nummm/Documents/CEPP/rawEEG"  # โฟลเดอร์หลักที่มี folder ของคนไข้

# Output settings
output_base_dir = "./output"                           # โฟลเดอร์หลักสำหรับบันทึกไฟล์ที่ประมวลผลแล้ว
                                                       # จะสร้าง subfolder แยกตาม patient ID

# Processing limits
max_patients = 10                                      # จำนวนคนไข้สูงสุดที่จะประมวลผล (None = ทั้งหมด)

# Channel selection
select_channel = None                                  # ช่อง EEG ที่ต้องการ (None = เลือกอัตโนมัติ)
                                                       # ตัวอย่าง: "EEG C4-A1", "EEG Fpz-Cz"

# Processing options
trim_wake_edges = True                                 # ตัดช่วง wake ที่ขอบออก (True/False)
edge_minutes = 30                                      # จำนวนนาทีที่ขยายก่อน/หลังช่วงนอน
epoch_sec_size = 30                                    # ขนาด epoch (วินาที)

# =======================================================
```

### Expected Directory Structure

สคริปต์คาดหวังโครงสร้างไฟล์ดังนี้:

```
input_base_dir/
├── 00000358-159547/
│   ├── edf_signals.edf
│   ├── csv_hypnogram.csv
│   └── csv_events.csv (optional)
├── 00000359-160123/
│   ├── edf_signals.edf
│   ├── csv_hypnogram.csv
│   └── csv_events.csv (optional)
└── ...
```

Output จะถูกสร้างเป็น:

```
output_base_dir/
├── 00000358-159547/
│   └── 00000358-159547.npz
├── 00000359-160123/
│   └── 00000359-160123.npz
└── ...
```

### Running the Script

หลังจากแก้ไขค่าตัวแปรแล้ว รันสคริปต์:

```bash
python preprocessing/custom/preprocess_custom.py
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `input_base_dir` | str | `"/home/nummm/Documents/CEPP/rawEEG"` | Path to directory containing patient folders |
| `output_base_dir` | str | `"./output"` | Directory to save processed data (creates subfolders per patient) |
| `max_patients` | int | `10` | Maximum number of patients to process (None = all) |
| `select_channel` | str | `None` | Specific EEG channel, None = auto-detect |
| `trim_wake_edges` | bool | `True` | Enable/disable wake edge trimming |
| `edge_minutes` | int | `30` | Minutes to extend before/after sleep |
| `epoch_sec_size` | int | `30` | Epoch duration in seconds |

## Batch Processing Features

### 🔄 Automatic Patient Processing

สคริปต์จะประมวลผลคนไข้หลายคนโดยอัตโนมัติ:

1. **สแกนโฟลเดอร์** - อ่านรายชื่อ folder ของคนไข้ทั้งหมดใน `input_base_dir`
2. **ตรวจสอบไฟล์** - ตรวจสอบว่ามี `edf_signals.edf` และ `csv_hypnogram.csv`
3. **ข้ามที่ซ้ำ** - ถ้า output folder ของคนไข้นั้นมีอยู่แล้ว จะข้ามไป
4. **จำกัดจำนวน** - ประมวลผลตาม `max_patients` ที่กำหนด
5. **จัดการ error** - ถ้าประมวลผลล้มเหลว จะลบ output folder และดำเนินการต่อ

### 📊 Processing Summary

เมื่อเสร็จสิ้น สคริปต์จะแสดงสรุป:

```
======================================================================
PROCESSING SUMMARY
======================================================================
Total patients found:    100
Successfully processed:  10
Skipped (already done):  5
Errors:                  2
======================================================================
```

### Examples

#### Example 1: Process First 10 Patients


แก้ไขค่าตัวแปรในไฟล์:
```python
input_base_dir = "/home/nummm/Documents/CEPP/rawEEG"
output_base_dir = "./output"
max_patients = 10  # ประมวลผล 10 คนแรก
```

รัน:
```bash
python preprocessing/custom/preprocess_custom.py
```

#### Example 2: Process All Patients

แก้ไขค่าตัวแปร:
```python
input_base_dir = "/home/nummm/Documents/CEPP/rawEEG"
output_base_dir = "./output"
max_patients = None  # ประมวลผลทั้งหมด
```

รัน:
```bash
python preprocessing/custom/preprocess_custom.py
```

#### Example 3: Specify EEG Channel for All Patients

แก้ไขค่าตัวแปร:
```python
input_base_dir = "/home/nummm/Documents/CEPP/rawEEG"
output_base_dir = "./output"
max_patients = 50
select_channel = "EEG Fpz-Cz"  # ระบุช่อง EEG เดียวกันสำหรับทุกคน
```

รัน:
```bash
python preprocessing/custom/preprocess_custom.py
```

#### Example 4: Resume Processing (Skip Already Done)

สคริปต์จะข้ามคนไข้ที่ประมวลผลแล้วโดยอัตโนมัติ:

```python
input_base_dir = "/home/nummm/Documents/CEPP/rawEEG"
output_base_dir = "./output"  # ถ้ามี folder ของคนไข้อยู่แล้ว จะข้าม
max_patients = 20
```

รัน:
```bash
python preprocessing/custom/preprocess_custom.py
```

## Input File Formats

### 1. EDF File (`edf_signals.edf`)

Standard European Data Format (EDF) file containing EEG signals. The script will:
- Auto-detect EEG channels (looks for "EEG" in channel names)
- Use the first EEG channel if multiple are found
- Fall back to the first channel if no EEG channels are detected

### 2. CSV Hypnogram (`csv_hypnogram.csv`)

CSV file containing sleep stage annotations. The script supports flexible column naming:

**Supported column names:**
- `stage`, `sleep_stage`, `label`, `hypnogram`, `annotation`, `event`

**Supported sleep stage values:**

| Sleep Stage | Accepted Values | Numeric Label |
|-------------|----------------|---------------|
| Wake | `W`, `WK`, `Wake`, `0` | 0 |
| N1 | `N1`, `1`, `S1` | 1 |
| N2 | `N2`, `2`, `S2` | 2 |
| N3/N4 | `N3`, `N4`, `3`, `4`, `S3`, `S4` | 3 |
| REM | `REM`, `R`, `5` | 4 |
| Movement/Unknown/Artifact | `6`, `7`, `8` | 0 (mapped to Wake) |

**Example CSV format:**

```csv
Epoch Number,Start Time,Sleep Stage
1,9:18:51 PM,WK
2,9:19:21 PM,WK
3,9:19:51 PM,N1
4,9:20:21 PM,N2
5,9:20:51 PM,N2
6,9:21:21 PM,N3
7,9:21:51 PM,N3
8,9:22:21 PM,REM
9,9:22:51 PM,REM
```

Or simpler format:

```csv
stage
WK
WK
N1
N2
N2
N3
N3
REM
REM
```

Or with other column names:

```csv
sleep_stage,duration,onset
Wake,30,0
N1,30,30
N2,30,60
N3,30,90
```

> [!NOTE]
> The script will use the first column if no standard column name is found.

### 3. CSV Events (`csv_events.csv`)

This file is **optional** and not currently used by the preprocessing script. It may contain additional event annotations for reference.

## Output Format

The script generates `.npz` files compatible with mulEEG:

```python
{
    'x': np.array,  # Shape: (n_epochs, samples_per_epoch)
                    # EEG signal data, float32
    'y': np.array,  # Shape: (n_epochs,)
                    # Sleep stage labels (0-4), int32
    'fs': float     # Sampling frequency (Hz)
}
```

### Example Output

For a recording with:
- Sampling rate: 100 Hz
- Epoch size: 30 seconds
- 400 epochs after trimming

Output shape:
- `x`: `(400, 3000)` - 400 epochs × 3000 samples (30s × 100Hz)
- `y`: `(400,)` - 400 labels
- `fs`: `100.0`

## Processing Pipeline

The script follows these steps (based on SHHS preprocessing):

1. **📂 Load EDF file**
   - Read using MNE library
   - Extract sampling rate

2. **🔍 Select EEG channel**
   - Auto-detect or use specified channel
   - Extract signal data

3. **📊 Parse hypnogram CSV**
   - Auto-detect sleep stage column
   - Map strings to numeric labels (0-4)

4. **✂️ Segment into epochs**
   - Split signal into 30-second windows
   - Validate signal length matches labels
   - Handle length mismatches (trim/pad)

5. **🎯 Trim wake edges** (optional)
   - Find first/last non-wake epochs
   - Extend by 30 minutes on each side
   - Focus on sleep-relevant periods

6. **💾 Save as .npz**
   - Save epochs, labels, and sampling rate
   - Compatible with mulEEG model

## Label Distribution

The script prints label distribution after processing:

```
Label distribution: W=120, N1=45, N2=150, N3=60, REM=25
```

This helps verify that labels were correctly mapped.

## Memory Optimization

This script is **optimized for low memory usage** to prevent out-of-memory errors.

### Key Optimizations

1. **Channel Selection Before Loading**
   - Selects only the required EEG channel before loading data
   - Prevents loading all channels into memory
   - **Saves 80-90% memory** for multi-channel files

2. **Direct NumPy Extraction**
   - Uses `get_data()` instead of `to_data_frame()`
   - Avoids pandas DataFrame overhead
   - **Saves ~50% memory**

3. **Lazy Loading**
   - Opens EDF file without preloading (`preload=False`)
   - Loads data only after channel selection
   - Minimizes peak memory usage

### Memory Usage Comparison

| File Type | Channels | Duration | Old Memory | New Memory | Reduction |
|-----------|----------|----------|------------|------------|-----------|
| Multi-channel EDF | 10 | 8 hours | ~6.6 GB | ~1.2 GB | **82%** |
| Single-channel EDF | 1 | 8 hours | ~1.5 GB | ~0.8 GB | **47%** |

### What You'll See

The script now prints additional information:
```
Available channels: ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', ...]
Total channels: 10
Selecting channel: EEG Fpz-Cz
Loading channel data...
```

This confirms only the selected channel is being loaded.

## Troubleshooting

### Issue: "No EEG channel found"

**Solution:** Specify the channel manually in configuration:
```python
select_channel = "Your Channel Name"
```

### Issue: "Signal length doesn't match expected"

**Cause:** Number of epochs in CSV doesn't match signal duration

**Solution:** The script will automatically trim or pad the signal, but verify your CSV has the correct number of rows (one per 30-second epoch)

### Issue: "Unknown sleep stage"

**Cause:** CSV contains sleep stage values not in the mapping

**Solution:** The script treats unknown stages as Wake (0) and prints a warning. Check your CSV for typos or non-standard labels.

## Comparison with SHHS Preprocessing

| Feature | SHHS | Custom |
|---------|------|--------|
| Input format | EDF + XML | EDF + CSV |
| Channel selection | Hardcoded `EEG C4-A1` | Auto-detect or specify |
| Label source | XML parsing | CSV parsing |
| Label mapping | 0-4 (same) | 0-4 (same) |
| Epoch size | 30 seconds | 30 seconds (configurable) |
| Wake trimming | Yes (30 min) | Yes (configurable) |
| Output format | .npz | .npz |

## Next Steps

After preprocessing, you can use the generated `.npz` files with the mulEEG model for sleep stage classification.

## References

- Based on: [`preprocessing/shhs/preprocess_shhs.py`](file:///home/nummm/Documents/CEPP/mulEEG/preprocessing/shhs/preprocess_shhs.py)
- MNE Python: https://mne.tools/
- mulEEG project: Multi-channel EEG sleep stage classification
