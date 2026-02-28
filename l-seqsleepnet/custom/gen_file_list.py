import os
import glob
import scipy.io as sio

# --- Configuration ---
mat_path = "./mat/"
channels = ["eeg", "eog"]


def generate_single_list(channel_type):
    # กำหนด path สำหรับบันทึกไฟล์ .txt
    tf_path = f"./file_list/{channel_type}/"
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)

    # สร้าง list เดียวเพื่อ test pretrain
    filename = "test_list.txt"
    filepath = os.path.join(tf_path, filename)

    print(f"Generating single list for channel: {channel_type} -> {filepath}")

    # ค้นหาไฟล์ .mat ทั้งหมดของ channel นี้
    search_pattern = os.path.join(mat_path, f"*_{channel_type}.mat")
    mat_files = sorted(glob.glob(search_pattern))

    with open(filepath, "w") as f:
        for full_mat_path in mat_files:
            sname = os.path.basename(full_mat_path)

            # โหลด label เพื่อหาจำนวน sample
            mat_contents = sio.loadmat(full_mat_path)
            num_sample = mat_contents["label"].size

            # บันทึก Path (แบบ relative เหมือน MATLAB) และจำนวน sample โดยคั่นด้วย Tab
            line = f"../../mat/{sname}\t{num_sample}\n"
            f.write(line)


# --- Main Processing Loop ---
for ch in channels:
    generate_single_list(ch)

print("Done! Single list for pretrain testing generated successfully.")
