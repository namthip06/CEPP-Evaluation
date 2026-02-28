import os
import glob
import logging
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import spectrogram, windows

# Set up logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
# ใช้ Absolute Paths ที่อิงจากตำแหน่งของ script ถัดจากการเตรียมข้อมูลดิบ (prepare_raw_data.py)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "processed_data"))
OUTPUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "mat"))

fs = 100  # sampling frequency
win_size = 2  # Window 2 วินาที
overlap = 1  # Overlap 1 วินาที
# คำนวณ NFFT ให้เหมือน MATLAB (nextpow2)
nfft = int(2 ** np.ceil(np.log2(win_size * fs)))


def main():
    logging.info("=== Starting Data Preparation (Spectrogram) ===")

    if not os.path.exists(INPUT_DIR):
        logging.error(
            f"Input Directory '{INPUT_DIR}' does not exist. Please run prepare_raw_data.py first."
        )
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # รายชื่อไฟล์ .mat ทั้งหมดใน INPUT_DIR
    file_list = sorted(glob.glob(os.path.join(INPUT_DIR, "*_processed.mat")))

    if not file_list:
        logging.warning(f"No processed data found in {INPUT_DIR}.")
        return

    logging.info(f"Found {len(file_list)} files to process in {INPUT_DIR}.")

    # สำหรับจัดการรันเลข Subject ใหม่ (Subject Numbering)
    unique_subs = {}

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        # ดึง Folder/id จากชื่อไฟล์ เช่น 01_processed.mat -> 01
        subject_id = file_name.replace("_processed.mat", "")

        # ลอจิกจำลอง sub != cur_sub โดยจับคู่ subject_id เดิมกับตัวเลขใหม่เป็นระเบียบ
        if subject_id not in unique_subs:
            unique_subs[subject_id] = len(unique_subs) + 1

        cur_sub_num = unique_subs[subject_id]
        night = 1  # ภายใต้โครงสร้าง rawEEG/[id]/edf_signals.edf สมมติว่าเป็นคืนที่ 1 เสมอ

        logging.debug(
            f"Processing {file_name} -> n{cur_sub_num:02d}_{night} (Original Subject: {subject_id})"
        )

        # Load ข้อมูลนำเข้าจากไฟล์ .mat (prepare_raw_data.py output)
        try:
            mat_contents = loadmat(file_path)
            data = mat_contents["data"]  # (N, 3000, 2)
            # แปลง label และ y ให้เป็น single (np.float32) ตามที่กำหนด
            y = mat_contents["y"].astype(np.float32)
            label = mat_contents["label"].astype(np.float32)
        except Exception as e:
            logging.error(f"Failed to load {file_name}: {e}")
            continue

        # วนลูปประมวลผลแยกแต่ละช่องสัญญาณ (EEG และ EOG)
        channels = ["eeg", "eog"]
        for idx, ch_name in enumerate(channels):
            # X1: Raw signals (Time domain) ในรูปแบบ Single Precision (ประหยัดแรม) มิติ (N, 3000)
            X1 = data[:, :, idx].astype(np.float32)

            # คำนวณ Spectrogram (STFT)
            # ใช้แบบ Vectorized โดยให้กระทำบนแกนสุดท้าย (axis=-1) ของ X1
            # Sxx จะมีมิติ (N, Frequency_bins, Time_bins) สำหรับ scipy.signal.spectrogram
            f, t, Sxx = spectrogram(
                X1,
                fs=fs,
                window=windows.hamming(win_size * fs),
                nperseg=win_size * fs,
                noverlap=overlap * fs,
                nfft=nfft,
                mode="magnitude",
                axis=-1,
            )

            # Log magnitude spectrum (20*log10(abs(Xk)))
            # ใส่ +1e-12 เพื่อป้องกันการหาค่า log(0)
            Xk = 20 * np.log10(Sxx + 1e-12)

            # การ Transpose เพื่อให้เหมือน MATLAB (Time bins x Frequency bins)
            # np.transpose(Xk, (0, 2, 1)) เปลี่ยนรูปจาก (N, 129, 29) -> (N, 29, 129)
            X2 = np.transpose(Xk, (0, 2, 1)).astype(np.float32)

            # บันทึกรูปแบบ Dual-stream ลงในไฟล์ .mat ใหม่
            # output format: n01_1_eeg.mat, n01_1_eog.mat, ...
            output_name = f"n{cur_sub_num:02d}_{night}_{ch_name}.mat"
            save_path = os.path.join(OUTPUT_DIR, output_name)

            try:
                savemat(
                    save_path,
                    {"X1": X1, "X2": X2, "label": label, "y": y},
                    do_compression=True,
                )
                logging.debug(
                    f"Saved: {output_name} | X1 shape: {X1.shape} | X2 shape: {X2.shape}"
                )
            except Exception as e:
                logging.error(f"Failed to save {output_name}: {e}")

    logging.info("=== Pipeline Execution Summary ===")
    logging.info(
        f"Processed {len(unique_subs)} subject(s). Outputs are saved in {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
