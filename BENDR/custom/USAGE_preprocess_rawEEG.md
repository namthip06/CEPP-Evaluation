# คู่มือการใช้งาน preprocess_rawEEG.py

## ภาพรวม
สคริปต์นี้ใช้สำหรับ preprocess ข้อมูล EEG จากโฟลเดอร์ rawEEG ที่มีโครงสร้างเฉพาะ

## โครงสร้างข้อมูล Input

```
/home/nummm/Documents/CEPP/rawEEG/
├── [id1]/
│   ├── csv_events.csv
│   ├── csv_hypnogram.csv
│   └── edf_signals.edf
├── [id2]/
│   ├── csv_events.csv
│   ├── csv_hypnogram.csv
│   └── edf_signals.edf
└── ... (100 folders)
```

### ไฟล์ csv_hypnogram.csv
```csv
Epoch Number,Start Time,Sleep Stage
1,9:18:51 PM,WK
2,9:19:21 PM,N1
3,9:19:51 PM,N2
```

## Output

ไฟล์ที่ประมวลผลแล้วจะถูกบันทึกที่:
```
/home/nummm/Documents/CEPP/BENDR/custom/preprocessing_output/
├── [id1]_preprocessed.fif
├── [id2]_preprocessed.fif
└── ...
```

## การทำงานของสคริปต์

### 1. การประมวลผลแต่ละ Subject
- อ่านไฟล์ `edf_signals.edf`
- เลือกเฉพาะช่อง EEG
- กรองสัญญาณ (band-pass 0.5-40 Hz, notch 50/60 Hz)
- Resample เป็น 256 Hz
- ตรวจสอบและแก้ไข bad channels
- อ่าน `csv_hypnogram.csv` และสร้าง annotations
- บันทึกเป็นไฟล์ `.fif`

### 2. การจัดการ Memory
- ประมวลผลทีละ subject (sequential, ไม่ใช้ batch/parallel)
- ลบข้อมูลออกจาก memory หลังประมวลผลแต่ละ subject
- เหมาะสำหรับระบบที่มี RAM น้อย

### 3. การจัดการ Error
- ข้ามโฟลเดอร์ที่ไม่มีไฟล์ `edf_signals.edf`
- แสดงข้อความ error แต่ไม่หยุดการทำงาน
- สรุปผลการประมวลผลตอนจบ

## วิธีใช้งาน

### ขั้นตอนที่ 1: ตรวจสอบข้อมูล
ตรวจสอบว่ามีโฟลเดอร์ rawEEG และมีข้อมูลครบถ้วน:
```bash
ls /home/nummm/Documents/CEPP/rawEEG/
```

### ขั้นตอนที่ 2: รันสคริปต์
```bash
cd /home/nummm/Documents/CEPP/BENDR/custom
python preprocess_rawEEG.py
```

### ขั้นตอนที่ 3: รอการประมวลผล
สคริปต์จะแสดงความคืบหน้าแบบนี้:
```
======================================================================
Progress: 1/100
======================================================================

Processing subject: id001
  EDF file: /home/nummm/Documents/CEPP/rawEEG/id001/edf_signals.edf
  Loading EDF file...
  Original sampling rate: 200.0 Hz
  Duration: 28800.0 seconds (480.0 minutes)
  Channels: 6
  Selected 6 EEG channels
  Applying filters...
  Resampling to 256 Hz...
  Checking for bad channels...
  No bad channels detected
  Parsing hypnogram...
  Parsed 960 sleep stage annotations
  Added sleep stage annotations
  Saving to: /home/nummm/Documents/CEPP/BENDR/custom/preprocessing_output/id001_preprocessed.fif
  ✓ Successfully processed id001
```

### ขั้นตอนที่ 4: ตรวจสอบผลลัพธ์
```bash
ls -lh /home/nummm/Documents/CEPP/BENDR/custom/preprocessing_output/
```

## การแมป Sleep Stages

สคริปต์จะแปลง sleep stage จาก CSV เป็นรูปแบบมาตรฐาน:

| CSV Value | Standard Format |
|-----------|-----------------|
| WK, W, WAKE | Sleep stage W |
| N1, 1 | Sleep stage 1 |
| N2, 2 | Sleep stage 2 |
| N3, 3 | Sleep stage 3 |
| N4, 4 | Sleep stage 4 |
| REM, R | Sleep stage R |

## พารามิเตอร์การประมวลผล

สคริปต์ใช้พารามิเตอร์ตามมาตรฐาน BENDR:

- **Band-pass filter**: 0.5-40 Hz
- **Notch filter**: 50, 60 Hz (power line noise)
- **Target sampling rate**: 256 Hz
- **Epoch duration**: 30 seconds (จาก hypnogram)
- **Bad channel threshold**: 5x median standard deviation

## การแก้ไขปัญหา

### ปัญหา: ไม่พบไฟล์ edf_signals.edf
```
Skipping id001: edf_signals.edf not found
```
**วิธีแก้**: ตรวจสอบว่าโฟลเดอร์มีไฟล์ครบถ้วน หรือสคริปต์จะข้ามโฟลเดอร์นี้โดยอัตโนมัติ

### ปัญหา: Memory Error
```
MemoryError: Unable to allocate array
```
**วิธีแก้**: สคริปต์ออกแบบมาสำหรับ low RAM แล้ว แต่ถ้ายังมีปัญหา:
1. ปิดโปรแกรมอื่นๆ
2. ประมวลผลทีละน้อย (แก้ไขโค้ดให้ประมวลผลเฉพาะบาง ID)

### ปัญหา: Hypnogram parsing error
```
Warning: Could not find required columns in csv_hypnogram.csv
```
**วิธีแก้**: ตรวจสอบว่า CSV มี columns: `Epoch Number`, `Start Time`, `Sleep Stage`

### ปัญหา: Filtering failed
```
Warning: Filtering failed
```
**วิธีแก้**: ข้อมูลอาจมีปัญหา แต่สคริปต์จะดำเนินการต่อโดยไม่กรอง

## สรุปผลการประมวลผล

เมื่อเสร็จสิ้น สคริปต์จะแสดงสรุป:
```
======================================================================
PROCESSING COMPLETE
======================================================================
Total subjects: 100
Successfully processed: 95
Skipped (no EDF): 3
Errors: 2
======================================================================

Preprocessed files saved to: /home/nummm/Documents/CEPP/BENDR/custom/preprocessing_output
Total output files: 95
```

## ขั้นตอนถัดไป

หลังจาก preprocess เสร็จแล้ว สามารถนำไฟล์ไป evaluate ด้วย BENDR:

1. แก้ไข `configs/downstream_datasets.yml` เพิ่ม dataset ใหม่:
```yaml
custom-rawEEG:
  name: "Custom RawEEG Dataset"
  toplevel: /home/nummm/Documents/CEPP/BENDR/custom/preprocessing_output/
  tmin: 0
  tlen: 30
  data_max: 100  # ปรับตามข้อมูลจริง
  data_min: -100
  extensions:
    - .fif
  picks:
    - eeg
  events:
    'Sleep stage W': 0
    'Sleep stage 1': 1
    'Sleep stage 2': 2
    'Sleep stage 3': 3
    'Sleep stage 4': 3
    'Sleep stage R': 4
```

2. รัน evaluation:
```bash
python downstream.py BENDR --ds-config configs/downstream.yml
```

## หมายเหตุ

- สคริปต์ใช้ absolute paths ทั้งหมด
- ไม่ใช้ batch หรือ parallel processing (เหมาะกับ low RAM)
- ประมวลผลทุก ID ใน rawEEG folder โดยอัตโนมัติ
- ข้าม ID ที่ไม่มี edf_signals.edf
- เวลาประมวลผล: ประมาณ 1-2 นาทีต่อ subject (ขึ้นกับขนาดไฟล์)
