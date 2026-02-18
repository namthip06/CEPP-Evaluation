# คู่มือการใช้งาน Sleep Staging Evaluation

## ภาพรวม
สคริปต์นี้ใช้สำหรับประเมินผล (evaluate) การทำนาย sleep stages โดยใช้โมเดล BENDR ที่เทรนสำเร็จแล้ว

## ข้อกำหนดเบื้องต้น

### 1. ข้อมูลที่ Preprocess แล้ว
ต้องรันสคริปต์ `preprocess_rawEEG.py` ก่อน เพื่อสร้างไฟล์ FIF ที่ประมวลผลแล้ว:
```bash
python custom/preprocess_rawEEG.py
```

ไฟล์จะถูกบันทึกที่: `custom/preprocessing_output/`

### 2. Pre-trained Weights
ดาวน์โหลดน้ำหนักที่เทรนสำเร็จแล้วจาก [GitHub Release v0.1-alpha](https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha):

```bash
# สร้างโฟลเดอร์สำหรับเก็บ weights
mkdir -p /home/nummm/Documents/CEPP/BENDR/weights

# ดาวน์โหลด (ตัวอย่าง - ปรับ URL ตามจริง)
cd /home/nummm/Documents/CEPP/BENDR/weights
wget https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt
wget https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/contextualizer.pt
```

ตรวจสอบว่ามีไฟล์:
- `/home/nummm/Documents/CEPP/BENDR/weights/encoder.pt`
- `/home/nummm/Documents/CEPP/BENDR/weights/contextualizer.pt`

### 3. Python Dependencies
```bash
pip install torch mne pandas numpy scikit-learn matplotlib seaborn
```

## วิธีใช้งาน

### ขั้นตอนที่ 1: ตรวจสอบข้อมูล
```bash
# ตรวจสอบว่ามีไฟล์ preprocessed
ls custom/preprocessing_output/*.fif | wc -l

# ตรวจสอบว่ามี weights
ls -lh /home/nummm/Documents/CEPP/BENDR/weights/
```

### ขั้นตอนที่ 2: รันการประเมินผล
```bash
cd /home/nummm/Documents/CEPP/BENDR
python custom/evaluate_sleep_staging.py
```

### ขั้นตอนที่ 3: รอการประมวลผล
สคริปต์จะแสดงความคืบหน้าแบบนี้:

```
======================================================================
BENDR Sleep Staging Evaluation
======================================================================
Preprocessed data: /home/nummm/Documents/CEPP/BENDR/custom/preprocessing_output
Output directory: /home/nummm/Documents/CEPP/BENDR/custom/evaluation_results
Encoder weights: /home/nummm/Documents/CEPP/BENDR/weights/encoder.pt
Context weights: /home/nummm/Documents/CEPP/BENDR/weights/contextualizer.pt
======================================================================

Found 100 preprocessed files
Using device: cpu

Loading BENDR model...
  Loading encoder from: /home/nummm/Documents/CEPP/BENDR/weights/encoder.pt
  Loading contextualizer from: /home/nummm/Documents/CEPP/BENDR/weights/contextualizer.pt
  Model loaded successfully on cpu

======================================================================
Progress: 1/100
======================================================================
Processing subject: id001
  Loading data from: .../preprocessing_output/id001_preprocessed.fif
  Loaded 960 epochs with 6 channels
  Making predictions...
  Calculating metrics...
  Saved predictions to: .../evaluation_results/id001_predictions.csv
  Saved metrics to: .../evaluation_results/id001_metrics.json
  ✓ Accuracy: 0.7854 (960 epochs)

[... continues for all 100 subjects ...]

Generating summary report...
  Saved summary to: .../evaluation_results/summary_report.csv
  Saved statistics to: .../evaluation_results/overall_statistics.json
  Saved confusion matrix to: .../evaluation_results/confusion_matrix.png

======================================================================
EVALUATION COMPLETE
======================================================================
Total files: 100
Successfully processed: 95
Errors: 5

Overall Statistics:
  Total epochs: 91200
  Mean accuracy: 0.7234
  Std accuracy: 0.0856
  Min accuracy: 0.5123
  Max accuracy: 0.8945
======================================================================

Results saved to: /home/nummm/Documents/CEPP/BENDR/custom/evaluation_results
```

## ผลลัพธ์ที่ได้

### โครงสร้างไฟล์ Output
```
custom/evaluation_results/
├── id001_predictions.csv       # การทำนายแต่ละ epoch
├── id001_metrics.json          # metrics ของ subject นี้
├── id002_predictions.csv
├── id002_metrics.json
├── ...
├── summary_report.csv          # สรุปผลทุก subjects
├── overall_statistics.json     # สถิติรวม
└── confusion_matrix.png        # confusion matrix รวม
```

### 1. Predictions CSV (`[subject_id]_predictions.csv`)
ไฟล์ CSV ที่มีการทำนายแต่ละ epoch:

| epoch | true_stage | predicted_stage | true_label | predicted_label | correct | prob_Wake | prob_N1 | prob_N2 | prob_N3 | prob_REM |
|-------|------------|-----------------|------------|-----------------|---------|-----------|---------|---------|---------|----------|
| 0 | Wake | Wake | 0 | 0 | True | 0.92 | 0.03 | 0.02 | 0.02 | 0.01 |
| 1 | N1 | N1 | 1 | 1 | True | 0.05 | 0.78 | 0.12 | 0.03 | 0.02 |
| 2 | N2 | N2 | 2 | 2 | True | 0.02 | 0.08 | 0.85 | 0.04 | 0.01 |

### 2. Metrics JSON (`[subject_id]_metrics.json`)
```json
{
  "subject_id": "id001",
  "n_epochs": 960,
  "overall_accuracy": 0.7854,
  "per_class_metrics": {
    "Wake": {
      "precision": 0.8234,
      "recall": 0.7891,
      "f1_score": 0.8059,
      "support": 245
    },
    "N1": {
      "precision": 0.6543,
      "recall": 0.6234,
      "f1_score": 0.6385,
      "support": 123
    },
    "N2": {
      "precision": 0.8123,
      "recall": 0.8456,
      "f1_score": 0.8286,
      "support": 387
    },
    "N3": {
      "precision": 0.7654,
      "recall": 0.7234,
      "f1_score": 0.7438,
      "support": 156
    },
    "REM": {
      "precision": 0.7891,
      "recall": 0.8123,
      "f1_score": 0.8005,
      "support": 49
    }
  },
  "confusion_matrix": [[...]]
}
```

### 3. Summary Report (`summary_report.csv`)
สรุปผลทุก subjects:

| subject_id | n_epochs | accuracy | f1_Wake | f1_N1 | f1_N2 | f1_N3 | f1_REM | support_Wake | support_N1 | ... |
|------------|----------|----------|---------|-------|-------|-------|--------|--------------|------------|-----|
| id001 | 960 | 0.7854 | 0.8059 | 0.6385 | 0.8286 | 0.7438 | 0.8005 | 245 | 123 | ... |
| id002 | 945 | 0.7123 | 0.7654 | 0.6123 | 0.7891 | 0.7234 | 0.7654 | 234 | 145 | ... |

### 4. Overall Statistics (`overall_statistics.json`)
```json
{
  "total_subjects": 95,
  "total_epochs": 91200,
  "mean_accuracy": 0.7234,
  "std_accuracy": 0.0856,
  "min_accuracy": 0.5123,
  "max_accuracy": 0.8945
}
```

### 5. Confusion Matrix (`confusion_matrix.png`)
รูปภาพ heatmap แสดง confusion matrix รวมจากทุก subjects

## การตีความผลลัพธ์

### Accuracy
- **> 0.80**: ดีมาก
- **0.70 - 0.80**: ดี
- **0.60 - 0.70**: พอใช้
- **< 0.60**: ต้องปรับปรุง

### Per-Class Metrics
- **Precision**: ความแม่นยำของการทำนาย (ทำนายถูกกี่เปอร์เซ็นต์)
- **Recall**: ความครอบคลุม (จับได้กี่เปอร์เซ็นต์ของ stage จริง)
- **F1-Score**: ค่าเฉลี่ยฮาร์มอนิกของ precision และ recall

### Sleep Stages
- **Wake (0)**: ตื่น
- **N1 (1)**: Non-REM Stage 1 (นอนตื้น)
- **N2 (2)**: Non-REM Stage 2 (นอนปานกลาง)
- **N3 (3)**: Non-REM Stage 3 (นอนลึก)
- **REM (4)**: Rapid Eye Movement (ฝัน)

## การแก้ไขปัญหา

### ปัญหา: ไม่พบ weights
```
Error: Encoder weights not found: /home/nummm/Documents/CEPP/BENDR/weights/encoder.pt
```
**วิธีแก้**: ดาวน์โหลด weights จาก GitHub และวางไว้ที่ path ที่ถูกต้อง

### ปัญหา: ไม่พบข้อมูล preprocessed
```
Error: No preprocessed FIF files found
```
**วิธีแก้**: รัน `preprocess_rawEEG.py` ก่อน

### ปัญหา: Memory Error
```
RuntimeError: CUDA out of memory
```
**วิธีแก้**: สคริปต์ใช้ `batch_size=1` อยู่แล้ว ถ้ายังมีปัญหา:
1. ปิดโปรแกรมอื่นๆ
2. ใช้ CPU แทน GPU (สคริปต์จะเลือกอัตโนมัติ)

### ปัญหา: Accuracy ต่ำมาก (< 0.3)
**สาเหตุที่เป็นไปได้**:
1. Weights ไม่ถูกต้องหรือเสียหาย
2. ข้อมูล preprocessing ไม่ถูกต้อง
3. Channel mapping ไม่ตรง

**วิธีแก้**:
1. ตรวจสอบ weights ใหม่
2. ตรวจสอบ preprocessing parameters
3. ตรวจสอบ channel names ในข้อมูล

## การปรับแต่ง

### เปลี่ยน Batch Size (ถ้ามี RAM มากพอ)
แก้ไขในไฟล์ `evaluate_sleep_staging.py`:
```python
# บรรทัดที่ 41
batch_size = 4  # เพิ่มจาก 1 เป็น 4
```

### เปลี่ยน Output Directory
แก้ไขในไฟล์ `evaluate_sleep_staging.py`:
```python
# บรรทัดที่ 38
OUTPUT_DIR = '/path/to/your/output'
```

### ประมวลผลเฉพาะบาง Subjects
แก้ไขในฟังก์ชัน `main()`:
```python
# กรองเฉพาะ subjects ที่ต้องการ
fif_files = [f for f in fif_files if f.startswith('id001') or f.startswith('id002')]
```

## หมายเหตุ

- สคริปต์ใช้ **inference เท่านั้น** (ไม่มีการ train/fine-tune)
- ประมวลผลแบบ sequential (ทีละ subject) เพื่อประหยัด RAM
- รองรับทั้ง CPU และ GPU (เลือกอัตโนมัติ)
- เวลาประมวลผล: ประมาณ 30-60 วินาทีต่อ subject (ขึ้นกับ CPU/GPU)
- ผลลัพธ์จะถูกบันทึกทันทีหลังประมวลผลแต่ละ subject

## ขั้นตอนถัดไป

หลังจากได้ผลลัพธ์แล้ว สามารถ:
1. วิเคราะห์ confusion matrix เพื่อดูว่า stage ไหนทำนายผิดบ่อย
2. ตรวจสอบ subjects ที่มี accuracy ต่ำ
3. เปรียบเทียบผลกับงานวิจัยอื่นๆ
4. ใช้ผลลัพธ์ในการปรับปรุงโมเดลหรือข้อมูล

## อ้างอิง

- [BENDR Paper](https://arxiv.org/pdf/2101.12037.pdf)
- [BENDR GitHub](https://github.com/SPOClab-ca/BENDR)
- [DN3 Documentation](https://dn3.readthedocs.io/)
