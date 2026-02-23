#!/usr/bin/env python3
"""
Clinical Processing Pipeline — custom/main.py
==============================================
End-to-end per-subject pipeline:
  EDF → HDF5 → Embeddings → Sleep Staging → Disease Prediction

Designed for ultra-low RAM usage: loads one subject at a time, deletes
intermediate files and tensors after each phase.
"""

# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: cuDNN Library Preload  —  MUST run BEFORE `import torch`
# ═══════════════════════════════════════════════════════════════════════════
import ctypes
import os
import sys

# ปิด nvfuser เพื่อลด warning / ป้องกัน hang
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"

# ค้นหา cuDNN .so ใน venv แล้ว preload ด้วย ctypes
_VENV_CUDNN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".venv", "lib", "python3.10", "site-packages", "nvidia", "cudnn", "lib",
)
if os.path.isdir(_VENV_CUDNN):
    # ต้อง load ตามลำดับ dependency
    for _lib in [
        "libcudnn_ops_infer.so.8",
        "libcudnn_ops_train.so.8",
        "libcudnn_cnn_infer.so.8",
        "libcudnn_cnn_train.so.8",
        "libcudnn_adv_infer.so.8",
        "libcudnn_adv_train.so.8",
        "libcudnn.so.8",
    ]:
        _path = os.path.join(_VENV_CUDNN, _lib)
        if os.path.isfile(_path):
            try:
                ctypes.cdll.LoadLibrary(_path)
            except OSError:
                pass  # ไม่ block ถ้า load ไม่ได้ — torch จะ fallback เอง

# ═══════════════════════════════════════════════════════════════════════════
# Normal imports  (torch is safe to import now)
# ═══════════════════════════════════════════════════════════════════════════
import gc
import json
import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — make sleepfm importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "sleepfm"))

from preprocessing.preprocessing import EDFToHDF5Converter
from models.models import (
    SetTransformer,
    SleepEventLSTMClassifier,
    DiagnosisFinetuneFullLSTMCOXPHWithDemo,
)
from models.dataset import SetTransformerDataset, collate_fn
from utils import load_config, load_data, save_data, count_parameters

# ═══════════════════════════════════════════════════════════════════════════
# 1. Logging Setup
# ═══════════════════════════════════════════════════════════════════════════

_ANSI_GREEN  = "\033[1;32m"
_ANSI_YELLOW = "\033[1;33m"
_ANSI_RED    = "\033[1;31m"
_ANSI_BLUE   = "\033[1;34m"
_ANSI_RESET  = "\033[0m"


class _ColouredFormatter(logging.Formatter):
    """Console formatter with ANSI level colours."""

    _LEVEL_COLOURS = {
        logging.DEBUG:    _ANSI_BLUE,
        logging.INFO:     _ANSI_GREEN,
        logging.WARNING:  _ANSI_YELLOW,
        logging.ERROR:    _ANSI_RED,
        logging.CRITICAL: _ANSI_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        colour = self._LEVEL_COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}[{record.levelname}]{_ANSI_RESET}"
        return super().format(record)


def setup_logging(log_path: str) -> logging.Logger:
    """Create a logger with console + file handlers."""
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler (coloured)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColouredFormatter("%(levelname)s %(message)s"))
    logger.addHandler(ch)

    # File handler (plain)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)
    return logger




# ═══════════════════════════════════════════════════════════════════════════
# 3. Helper: Subject Discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_subjects(data_dir: str, log: logging.Logger) -> List[Dict[str, str]]:
    """
    Scan data_dir for subject sub-folders.
    Each folder must contain  edf_signals.edf  +  csv_hypnogram.csv.
    Returns a list of dicts: {study_id, edf_path, csv_path}.
    """
    subjects = []

    # ── ลอง flat layout ก่อน (demo_psg.edf อยู่ตรงๆ ใน data_dir) ──
    flat_edfs = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".edf") and os.path.isfile(os.path.join(data_dir, f))
    )
    # ถ้าเจอ .edf ระดับ root → ใช้ flat mode (เหมือน demo)
    if flat_edfs and not any(
        os.path.isdir(os.path.join(data_dir, d)) and
        os.path.isfile(os.path.join(data_dir, d, "edf_signals.edf"))
        for d in os.listdir(data_dir)
    ):
        for edf_name in flat_edfs:
            sid = os.path.splitext(edf_name)[0]
            csv_path = os.path.join(data_dir, f"{sid}.csv")
            if not os.path.isfile(csv_path):
                log.error(f"Subject {sid} — missing CSV file. Skipping…")
                continue
            subjects.append({
                "study_id": sid,
                "edf_path": os.path.join(data_dir, edf_name),
                "csv_path": csv_path,
            })
        log.info(f"Discovered {len(subjects)} subject(s) [flat layout] in {data_dir}")
        return subjects

    # ── Sub-folder layout: {subject_id}/edf_signals.edf + csv_hypnogram.csv ──
    folders = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        edf_path = os.path.join(folder_path, "edf_signals.edf")
        csv_path = os.path.join(folder_path, "csv_hypnogram.csv")

        if not os.path.isfile(edf_path):
            log.error(f"Subject {folder} — missing edf_signals.edf. Skipping…")
            continue
        if not os.path.isfile(csv_path):
            log.error(f"Subject {folder} — missing csv_hypnogram.csv. Skipping…")
            continue

        subjects.append({
            "study_id": folder,
            "edf_path": edf_path,
            "csv_path": csv_path,
        })

    if not subjects:
        log.error(f"No valid subjects found in {data_dir}")
    else:
        log.info(f"Discovered {len(subjects)} subject(s) [sub-folder layout] in {data_dir}")
    return subjects


# ── Stage mapping: รองรับทั้ง demo format และ real data format ──────────────
# Demo CSV : StageName ∈ {Wake, Stage 1, Stage 2, Stage 3, REM, Unknown}
# Real CSV : Sleep Stage ∈ {WK, N1, N2, N3, REM, NS}
STAGE_MAP = {
    # demo format
    "Wake": 0, "Stage 1": 1, "Stage 2": 2, "Stage 3": 3, "REM": 4,
    # real data format
    "WK": 0, "W": 0, "N1": 1, "N2": 2, "N3": 3,
}
NS_LABELS = {"Unknown", "NS"}  # labels ที่ถือว่าเป็น Not Scored
EPOCH_5SEC = 5                 # model expects 5-sec epochs


def _detect_csv_format(df: pd.DataFrame) -> str:
    """Return 'demo' or 'real' based on column names."""
    cols = {c.strip() for c in df.columns}
    if "StageName" in cols or "StageNumber" in cols:
        return "demo"
    if "Sleep Stage" in cols:
        return "real"
    raise ValueError(f"Unrecognised hypnogram CSV columns: {df.columns.tolist()}")


def validate_hypnogram(
    csv_path: str,
    study_id: str,
    min_minutes: float,
    ns_threshold: float,
    log: logging.Logger,
) -> Optional[pd.DataFrame]:
    """
    Load hypnogram CSV (demo or real format), filter out NS epochs,
    validate length, and return a cleaned DataFrame with columns:
        Start, Stop, StageName, StageNumber, EmbeddingNumber
    in 5-second epoch resolution.  Returns None to skip this subject.
    """
    # ── Read with BOM handling ────────────────────────────────────────
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]  # strip whitespace
    fmt = _detect_csv_format(df)

    # ── Normalise to a common (stage_name) column ─────────────────────
    if fmt == "demo":
        df["_stage"] = df["StageName"].str.strip()
        epoch_sec = EPOCH_5SEC  # demo already 5-sec
    else:  # real
        df["_stage"] = df["Sleep Stage"].str.strip()
        epoch_sec = 30  # real PSG = 30-sec epochs

    total_epochs = len(df)
    total_minutes = total_epochs * epoch_sec / 60.0

    # ── NS counting ───────────────────────────────────────────────────
    ns_mask = df["_stage"].isin(NS_LABELS)
    # demo format อาจมี StageNumber == -1 ด้วย
    if "StageNumber" in df.columns:
        ns_mask = ns_mask | (df["StageNumber"] == -1)
    ns_count = int(ns_mask.sum())
    ns_pct = ns_count / max(total_epochs, 1)

    log.info(
        f"  Hypnogram ({fmt}, {epoch_sec}s epochs): "
        f"{total_epochs} epochs ({total_minutes:.1f} min), "
        f"NS={ns_count} ({ns_pct*100:.1f}%)"
    )

    if ns_pct > ns_threshold:
        log.warning(
            f"  NS fraction {ns_pct*100:.1f}% > threshold "
            f"{ns_threshold*100:.0f}%. Skipping {study_id}."
        )
        return None

    # ── Remove NS rows ────────────────────────────────────────────────
    df_clean = df[~ns_mask].copy()
    remaining = len(df_clean)
    remaining_min = remaining * epoch_sec / 60.0

    log.info(
        f"  After NS removal: {remaining} epochs ({remaining_min:.1f} min), "
        f"removed {ns_count} epochs"
    )

    if remaining_min < min_minutes:
        log.warning(
            f"  Recording too short ({remaining_min:.1f} min < "
            f"{min_minutes} min). Skipping {study_id}."
        )
        return None

    # ── Map stage names → numbers ─────────────────────────────────────
    df_clean["_num"] = df_clean["_stage"].map(STAGE_MAP)
    unmapped = df_clean["_num"].isna().sum()
    if unmapped > 0:
        bad = df_clean.loc[df_clean["_num"].isna(), "_stage"].unique().tolist()
        log.warning(f"  {unmapped} epoch(s) with unrecognised stage {bad} — dropped")
        df_clean = df_clean.dropna(subset=["_num"])
    df_clean["_num"] = df_clean["_num"].astype(int)

    # ── Build output DF in model's expected format ────────────────────
    # Model expects 5-sec epochs with columns:
    #   Start, Stop, StageName, StageNumber, EmbeddingNumber
    out_rows = []
    emb_idx = 0
    inv_map = {0: "Wake", 1: "Stage 1", 2: "Stage 2", 3: "Stage 3", 4: "REM"}

    for _, row in df_clean.iterrows():
        stage_num = int(row["_num"])
        stage_name = inv_map[stage_num]
        # จำนวน 5-sec sub-epochs ต่อ 1 epoch ต้นทาง
        n_sub = epoch_sec // EPOCH_5SEC
        for j in range(n_sub):
            start = emb_idx * EPOCH_5SEC
            stop = start + EPOCH_5SEC
            out_rows.append({
                "Start": start,
                "Stop": stop,
                "StageName": stage_name,
                "StageNumber": stage_num,
                "EmbeddingNumber": emb_idx,
            })
            emb_idx += 1

    out_df = pd.DataFrame(out_rows)
    log.info(f"  Output labels: {len(out_df)} rows (5-sec), "
             f"{len(out_df)*EPOCH_5SEC/60:.1f} min")
    return out_df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Helper: EDF → HDF5 + Signal Stats
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_edf(
    edf_path: str,
    hdf5_path: str,
    log: logging.Logger,
) -> bool:
    """Convert EDF to HDF5 and log per-channel stats. Returns True on success."""
    try:
        converter = EDFToHDF5Converter(
            root_dir="/dummy", target_dir="/dummy", resample_rate=128
        )
        converter.convert(edf_path, hdf5_path)
    except Exception as e:
        log.error(f"  EDF conversion failed: {e}")
        return False

    # Log signal stats
    with h5py.File(hdf5_path, "r") as f:
        for key in f.keys():
            data = f[key][:].astype(np.float32)  # avoid float16 overflow
            mn, mx = float(np.nanmin(data)), float(np.nanmax(data))
            mean_val = float(np.nanmean(data))
            has_nan = bool(np.isnan(data).any())
            has_inf = bool(np.isinf(data).any()) or np.isinf(mean_val)
            zeros_pct = (data == 0).sum() / max(data.size, 1) * 100

            status = "OK"
            if has_nan:
                status = "NaN detected!"
            elif has_inf:
                status = "Inf detected!"
            elif zeros_pct > 10:
                status = f"High zeros ({zeros_pct:.1f}%)"

            log.info(
                f"  Channel {key:20s} | shape={str(data.shape):16s} "
                f"| mean={mean_val:+.4f} | min={mn:.4f} | max={mx:.4f} "
                f"| zeros={zeros_pct:.1f}% | {status}"
            )

    return True


# ═══════════════════════════════════════════════════════════════════════════
# 6. Helper: Embedding Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_embeddings(
    hdf5_path: str,
    emb_dir: str,
    emb_5min_dir: str,
    study_id: str,
    model: nn.Module,
    config: dict,
    channel_groups: dict,
    device: torch.device,
    batch_size: int,
    log: logging.Logger,
) -> bool:
    """Generate and save 5-sec and 5-min embeddings. Returns True on success."""
    try:
        dataset = SetTransformerDataset(
            config, channel_groups, hdf5_paths=[hdf5_path], split="test"
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=1,
            shuffle=False, collate_fn=collate_fn,
        )
    except Exception as e:
        log.error(f"  Embedding dataset creation failed: {e}")
        return False

    modality_types = config["modality_types"]
    embed_dim = config["embed_dim"]

    emb_path = os.path.join(emb_dir, f"{study_id}.hdf5")
    emb_5min_path = os.path.join(emb_5min_dir, f"{study_id}.hdf5")

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_data, mask_list, file_paths, dset_names_list, chunk_starts = batch
            signals = [batch_data[i].to(device, dtype=torch.float) for i in range(4)]
            masks = [mask_list[i].to(device, dtype=torch.bool) for i in range(4)]

            embeddings = [model(signals[i], masks[i]) for i in range(4)]

            # 5-min aggregated embeddings (index [0])
            embeddings_5min = [e[0].unsqueeze(1) for e in embeddings]
            for i in range(len(file_paths)):
                chunk_start = chunk_starts[i]
                with h5py.File(emb_5min_path, "a") as hf:
                    for mod_idx, mod_type in enumerate(modality_types):
                        emb_data = embeddings_5min[mod_idx][i].cpu().numpy()
                        chunk_start_correct = chunk_start // (embed_dim * 5 * 60)
                        chunk_end = chunk_start_correct + emb_data.shape[0]
                        if mod_type in hf:
                            dset = hf[mod_type]
                            if dset.shape[0] < chunk_end:
                                dset.resize((chunk_end,) + emb_data.shape[1:])
                            dset[chunk_start_correct:chunk_end] = emb_data
                        else:
                            hf.create_dataset(
                                mod_type, data=emb_data,
                                chunks=(embed_dim,) + emb_data.shape[1:],
                                maxshape=(None,) + emb_data.shape[1:],
                            )

            # 5-sec granular embeddings (index [1])
            embeddings_5sec = [e[1] for e in embeddings]
            for i in range(len(file_paths)):
                chunk_start = chunk_starts[i]
                with h5py.File(emb_path, "a") as hf:
                    for mod_idx, mod_type in enumerate(modality_types):
                        emb_data = embeddings_5sec[mod_idx][i].cpu().numpy()
                        chunk_start_correct = chunk_start // (embed_dim * 5)
                        chunk_end = chunk_start_correct + emb_data.shape[0]
                        if mod_type in hf:
                            dset = hf[mod_type]
                            if dset.shape[0] < chunk_end:
                                dset.resize((chunk_end,) + emb_data.shape[1:])
                            dset[chunk_start_correct:chunk_end] = emb_data
                        else:
                            hf.create_dataset(
                                mod_type, data=emb_data,
                                chunks=(embed_dim,) + emb_data.shape[1:],
                                maxshape=(None,) + emb_data.shape[1:],
                            )

            # Free GPU memory
            del signals, masks, embeddings, embeddings_5min, embeddings_5sec
            torch.cuda.empty_cache()

    del dataset, dataloader
    gc.collect()

    log.info(f"  Embeddings saved → {emb_path}")
    log.info(f"  5-min agg saved  → {emb_5min_path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 7. Inline Datasets for Sleep Staging & Disease Prediction
#    (adapted from notebook — kept self-contained)
# ═══════════════════════════════════════════════════════════════════════════

class _SleepStagingDataset(Dataset):
    """Minimal sleep-staging dataset for a single subject."""

    def __init__(self, config, channel_groups, hdf5_path, label_path):
        self.config = config
        self.max_channels = config["max_channels"]
        self.context = int(config["context"])
        self.channel_like = config["channel_like"]
        self.max_seq_len = config["model_params"]["max_seq_length"]

        self.index_map = []
        if self.context == -1:
            self.index_map.append((hdf5_path, label_path, -1))
        else:
            with h5py.File(hdf5_path, "r") as hf:
                dset_names = list(hf.keys())
                if dset_names:
                    dataset_length = hf[dset_names[0]].shape[0]
                    for i in range(0, dataset_length, self.context):
                        self.index_map.append((hdf5_path, label_path, i))

        self.total_len = len(self.index_map)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        hdf5_path, label_path, start_index = self.index_map[idx]

        labels_df = pd.read_csv(label_path)
        labels_df["StageNumber"] = labels_df["StageNumber"].replace(-1, 0)
        y_data = labels_df["StageNumber"].to_numpy()
        if self.context != -1:
            y_data = y_data[start_index:start_index + self.context]

        x_data = []
        with h5py.File(hdf5_path, "r") as hf:
            for dname in hf.keys():
                if dname in self.channel_like:
                    if self.context == -1:
                        x_data.append(hf[dname][:])
                    else:
                        x_data.append(hf[dname][start_index:start_index + self.context])

        if not x_data:
            return self.__getitem__((idx + 1) % self.total_len)

        x_data = torch.tensor(np.array(x_data), dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)

        min_length = min(x_data.shape[1], len(y_data))
        x_data = x_data[:, :min_length, :]
        y_data = y_data[:min_length]

        return x_data, y_data, self.max_channels, self.max_seq_len, hdf5_path


def _sleep_staging_collate(batch):
    """Collate for sleep staging — mirrors notebook implementation."""
    x_data, y_data, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)
    num_channels = max(max_channels_list)

    max_seq_len_temp = max(item.size(1) for item in x_data)
    max_seq_len = (
        max_seq_len_temp if max_seq_len_list[0] is None
        else min(max_seq_len_temp, max_seq_len_list[0])
    )

    padded_x, padded_y, padded_mask = [], [], []

    for x_item, y_item in zip(x_data, y_data):
        tgt = np.where(y_item.numpy() > 0, 1, 0)
        moving_avg = np.convolve(tgt, np.ones(1080) / 1080, mode="valid")
        try:
            first_nz = np.where(moving_avg > 0.5)[0][0]
        except IndexError:
            first_nz = 0
        first_nz = max(first_nz, 0)

        c, s, e = x_item.size()
        c = min(c, num_channels)
        s = min(s, max_seq_len + first_nz)

        px = torch.zeros((num_channels, max_seq_len, e))
        mask = torch.ones((num_channels, max_seq_len))
        px[:c, :s - first_nz, :e] = x_item[:c, first_nz:s, :e]
        mask[:c, :s - first_nz] = 0

        py = torch.zeros(max_seq_len)
        py[:s - first_nz] = y_item[first_nz:s]

        padded_x.append(px)
        padded_y.append(py)
        padded_mask.append(mask)

    return (
        torch.stack(padded_x),
        torch.stack(padded_y),
        torch.stack(padded_mask),
        hdf5_path_list,
    )


class _DiagnosisDataset(Dataset):
    """Minimal diagnosis dataset for a single subject."""

    def __init__(self, config, channel_groups, hdf5_path, demo_labels_path, labels_path):
        self.config = config
        self.channel_groups = channel_groups
        self.max_channels = config["max_channels"]

        demo_df = pd.read_csv(demo_labels_path).set_index("Study ID")
        study_ids = set(demo_df.index)

        is_event_df = pd.read_csv(
            os.path.join(labels_path, "is_event.csv")
        ).set_index("Study ID")
        event_time_df = pd.read_csv(
            os.path.join(labels_path, "time_to_event.csv")
        ).set_index("Study ID")

        file_id = os.path.basename(hdf5_path).split(".")[0]
        if file_id not in study_ids:
            raise ValueError(f"Study ID '{file_id}' not found in demographic CSV")

        self.labels = {
            "is_event": list(is_event_df.loc[file_id].values),
            "event_time": list(event_time_df.loc[file_id].values),
            "demo_feats": list(demo_df.loc[file_id].values),
        }
        self.hdf5_path = hdf5_path
        self.max_seq_len = config["model_params"]["max_seq_length"]
        self.total_len = 1

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        x_data = []
        with h5py.File(self.hdf5_path, "r") as hf:
            for dname in hf.keys():
                if isinstance(hf[dname], h5py.Dataset) and dname in self.config["modality_types"]:
                    x_data.append(hf[dname][:])

        if not x_data:
            raise RuntimeError("No modality data found in embedding HDF5")

        x_data = torch.tensor(np.array(x_data), dtype=torch.float32)
        event_time = torch.tensor(self.labels["event_time"], dtype=torch.float32)
        is_event = torch.tensor(self.labels["is_event"])
        demo_feats = torch.tensor(self.labels["demo_feats"], dtype=torch.float32)

        return x_data, event_time, is_event, demo_feats, self.max_channels, self.max_seq_len, self.hdf5_path


def _diagnosis_collate(batch):
    """Collate for disease prediction — mirrors notebook implementation."""
    x_data, event_time, is_event, demo_feats, max_channels_list, max_seq_len_list, hdf5_path_list = zip(*batch)
    num_channels = max(max_channels_list)

    max_seq_len = (
        max(item.size(1) for item in x_data) if max_seq_len_list[0] is None
        else max_seq_len_list[0]
    )

    padded_x, padded_mask = [], []
    for item in x_data:
        c, s, e = item.size()
        c = min(c, num_channels)
        s = min(s, max_seq_len)
        px = torch.zeros((num_channels, max_seq_len, e))
        mask = torch.ones((num_channels, max_seq_len))
        px[:c, :s, :e] = item[:c, :s, :e]
        mask[:c, :s] = 0
        padded_x.append(px)
        padded_mask.append(mask)

    return (
        torch.stack(padded_x),
        torch.stack(event_time),
        torch.stack(is_event),
        torch.stack(demo_feats),
        torch.stack(padded_mask),
        hdf5_path_list,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. Model Loading (once at startup)
# ═══════════════════════════════════════════════════════════════════════════

def load_set_transformer(sleepfm_dir: str, device: torch.device, log: logging.Logger):
    """Load the pretrained SetTransformer (embedding model)."""
    model_path = os.path.join(sleepfm_dir, "checkpoints", "model_base")
    config_path = os.path.join(model_path, "config.json")
    channel_groups_path = os.path.join(sleepfm_dir, "configs", "channel_groups.json")

    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

    model = SetTransformer(
        config["in_channels"], config["patch_size"], config["embed_dim"],
        config["num_heads"], config["num_layers"],
        pooling_head=config["pooling_head"], dropout=0.0,
    )

    if device.type == "cuda":
        model = nn.DataParallel(model)
    model.to(device)

    ckpt = torch.load(os.path.join(model_path, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    layers, params = count_parameters(model)
    log.info(f"SetTransformer loaded: {params/1e6:.2f}M params, {layers} layers")
    return model, config, channel_groups


def load_sleep_staging_model(sleepfm_dir: str, device: torch.device, log: logging.Logger):
    """Load the finetuned sleep-staging model."""
    model_path = os.path.join(sleepfm_dir, "checkpoints", "model_sleep_staging")
    ss_config = load_data(os.path.join(model_path, "config.json"))

    ModelClass = globals().get(ss_config["model"]) or getattr(
        sys.modules[__name__], ss_config["model"]
    )
    model = ModelClass(**ss_config["model_params"]).to(device)
    model = nn.DataParallel(model)

    ckpt = torch.load(os.path.join(model_path, "best.pth"), map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    layers, params = count_parameters(model)
    log.info(f"SleepStaging model loaded: {params/1e6:.2f}M params, {layers} layers")
    return model, ss_config


def load_diagnosis_model(sleepfm_dir: str, device: torch.device, log: logging.Logger):
    """Load the finetuned disease-prediction model."""
    model_path = os.path.join(sleepfm_dir, "checkpoints", "model_diagnosis")
    dx_config = load_data(os.path.join(model_path, "config.json"))
    dx_config["model_params"]["dropout"] = 0.0

    ModelClass = globals().get(dx_config["model"]) or getattr(
        sys.modules[__name__], dx_config["model"]
    )
    model = ModelClass(**dx_config["model_params"]).to(device)
    model = nn.DataParallel(model)

    ckpt = torch.load(os.path.join(model_path, "best.pth"), map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    layers, params = count_parameters(model)
    log.info(f"Diagnosis model loaded: {params/1e6:.2f}M params, {layers} layers")
    return model, dx_config


# ═══════════════════════════════════════════════════════════════════════════
# 9. Per-Subject Inference Runners
# ═══════════════════════════════════════════════════════════════════════════

def run_sleep_staging(
    emb_path: str,
    label_path: str,
    study_id: str,
    model: nn.Module,
    ss_config: dict,
    channel_groups: dict,
    device: torch.device,
    output_dir: str,
    batch_size: int,
    log: logging.Logger,
) -> bool:
    """Run sleep-staging inference for one subject."""
    try:
        dataset = _SleepStagingDataset(
            ss_config, channel_groups, emb_path, label_path
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=1, collate_fn=_sleep_staging_collate,
        )
    except Exception as e:
        log.error(f"  Sleep staging dataset error: {e}")
        return False

    all_targets, all_outputs, all_masks = [], [], []

    model.eval()
    with torch.no_grad():
        for x, y, mask, _ in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            outputs, out_mask = model(x, mask)
            all_targets.append(y.cpu().numpy())
            all_outputs.append(torch.softmax(outputs, dim=-1).cpu().numpy())
            all_masks.append(out_mask.cpu().numpy())

    if not all_outputs:
        log.warning("  No sleep staging outputs produced")
        return False

    # Flatten and filter padding
    targets = np.concatenate([t.reshape(-1) for t in all_targets])
    outputs = np.concatenate([o.reshape(-1, o.shape[-1]) for o in all_outputs])
    masks = np.concatenate([m.reshape(-1) for m in all_masks])

    real_mask = masks == 0
    targets_real = targets[real_mask]
    outputs_real = outputs[real_mask]
    preds = np.argmax(outputs_real, axis=1)

    # Log stage distribution
    stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    counts = Counter(preds.tolist())
    dist_str = ", ".join(f"{stage_names.get(k, k)}={v}" for k, v in sorted(counts.items()))
    log.info(f"  Predicted stages: {dist_str}")

    # Save CSV
    save_dir = os.path.join(output_dir, "sleep_staging")
    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, f"{study_id}.csv")

    df = pd.DataFrame({
        "predicted_stage": preds,
        "true_stage": targets_real.astype(int),
        **{f"prob_{stage_names.get(i, i)}": outputs_real[:, i] for i in range(outputs_real.shape[1])},
    })
    df.to_csv(out_csv, index=False)
    log.info(f"  Sleep staging saved → {out_csv}")

    del dataset, loader, all_targets, all_outputs, all_masks
    gc.collect()
    torch.cuda.empty_cache()
    return True


def run_disease_prediction(
    emb_path: str,
    study_id: str,
    model: nn.Module,
    dx_config: dict,
    channel_groups: dict,
    device: torch.device,
    data_dir: str,
    output_dir: str,
    sleepfm_dir: str,
    batch_size: int,
    log: logging.Logger,
) -> bool:
    """Run disease-prediction inference for one subject."""
    demo_path = os.path.join(data_dir, "demo_age_gender.csv")
    if not os.path.isfile(demo_path):
        log.warning(f"  demo_age_gender.csv not found in {data_dir} — skipping diagnosis")
        return False

    try:
        dx_config_copy = dict(dx_config)
        dx_config_copy["labels_path"] = data_dir
        dataset = _DiagnosisDataset(
            dx_config_copy, channel_groups, emb_path, demo_path, data_dir
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=1, collate_fn=_diagnosis_collate,
        )
    except Exception as e:
        log.error(f"  Diagnosis dataset error: {e}")
        return False

    model.eval()
    all_outputs = []
    with torch.no_grad():
        for item in loader:
            x, evt, ise, demo, mask, _ = item
            x = x.to(device)
            demo = demo.to(device)
            mask = mask.to(device)
            outputs = model(x, mask, demo)
            all_outputs.append(outputs.cpu().numpy())

    if not all_outputs:
        log.warning("  No diagnosis outputs produced")
        return False

    outputs_arr = np.concatenate(all_outputs, axis=0)

    # Map to phecodes
    label_mapping_path = os.path.join(sleepfm_dir, "configs", "label_mapping.csv")
    if os.path.isfile(label_mapping_path):
        labels_df = pd.read_csv(label_mapping_path)
        labels_df["hazard_score"] = outputs_arr[0]
    else:
        labels_df = pd.DataFrame({"hazard_score": outputs_arr[0]})

    save_dir = os.path.join(output_dir, "diagnosis")
    os.makedirs(save_dir, exist_ok=True)
    out_csv = os.path.join(save_dir, f"{study_id}.csv")
    labels_df.to_csv(out_csv, index=False)
    log.info(f"  Diagnosis saved → {out_csv} ({len(labels_df)} phecodes)")

    del dataset, loader, all_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return True


# ═══════════════════════════════════════════════════════════════════════════
# 10. Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Configuration ─────────────────────────────────────────────────
    # แก้ไขค่าเหล่านี้ตามต้องการก่อนรัน
    DATA_DIR      = "/home/nummm/Documents/CEPP/rawEEG/"   # โฟลเดอร์ที่มีไฟล์ .edf + .csv
    OUTPUT_DIR    = "custom/results"                # โฟลเดอร์สำหรับบันทึกผลลัพธ์
    SLEEPFM_DIR   = "sleepfm"               # โฟลเดอร์ sleepfm package
    MIN_MINUTES   = 15.0                    # ความยาวขั้นต่ำ (นาที)
    NS_THRESHOLD  = 0.5                     # สัดส่วน Unknown/NS สูงสุดก่อน skip
    BATCH_SIZE    = 16                      # batch size สำหรับ inference
    DEVICE        = "auto"                  # "cuda", "cpu", หรือ "auto"
    # ─────────────────────────────────────────────────────────────────

    # Resolve paths
    data_dir = os.path.abspath(DATA_DIR)
    output_dir = os.path.abspath(OUTPUT_DIR)
    sleepfm_dir = os.path.abspath(SLEEPFM_DIR)
    min_minutes = MIN_MINUTES
    ns_threshold = NS_THRESHOLD
    batch_size = BATCH_SIZE

    os.makedirs(output_dir, exist_ok=True)
    log = setup_logging(os.path.join(output_dir, "pipeline.log"))

    log.info("=" * 60)
    log.info("SleepFM Clinical Processing Pipeline")
    log.info("=" * 60)
    log.info(f"  data_dir    : {data_dir}")
    log.info(f"  output_dir  : {output_dir}")
    log.info(f"  sleepfm_dir : {sleepfm_dir}")
    log.info(f"  min_minutes : {min_minutes}")
    log.info(f"  ns_threshold: {ns_threshold}")
    log.info(f"  batch_size  : {batch_size}")

    # Device
    if DEVICE == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(DEVICE)
    log.info(f"  device      : {device}")
    if device.type == "cuda":
        log.info(f"  GPU count   : {torch.cuda.device_count()}")
        log.info(f"  GPU name    : {torch.cuda.get_device_name(0)}")

    # ── Load all models once ──────────────────────────────────────────
    log.info("")
    log.info("Loading models…")
    emb_model, emb_config, channel_groups = load_set_transformer(sleepfm_dir, device, log)
    ss_model, ss_config = load_sleep_staging_model(sleepfm_dir, device, log)
    dx_model, dx_config = load_diagnosis_model(sleepfm_dir, device, log)
    log.info("All models loaded.\n")

    # ── Create output subdirectories ──────────────────────────────────
    tmp_dir       = os.path.join(output_dir, "tmp")
    emb_dir       = os.path.join(output_dir, "embeddings")
    emb_5min_dir  = os.path.join(output_dir, "embeddings_5min")
    for d in [tmp_dir, emb_dir, emb_5min_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Discover subjects ─────────────────────────────────────────────
    subjects = discover_subjects(data_dir, log)
    if not subjects:
        log.error("No subjects to process. Exiting.")
        return

    # ── Summary tracker ───────────────────────────────────────────────
    summary_rows: List[Dict[str, Any]] = []

    # ── Per-subject loop ──────────────────────────────────────────────
    for subj in tqdm(subjects, desc="Processing subjects"):
        sid = subj["study_id"]
        t0 = time.time()
        log.info(f"{'─' * 60}")
        log.info(f"Subject: {sid}")
        log.info(f"{'─' * 60}")

        row: Dict[str, Any] = {
            "Study ID": sid, "Status": "PENDING",
            "Channels": "-", "Epochs (orig)": "-",
            "Epochs (clean)": "-", "NS removed": "-",
            "Notes": "",
        }

        # Phase 3: Hypnogram validation
        hyp_clean = validate_hypnogram(
            subj["csv_path"], sid,
            min_minutes, ns_threshold, log,
        )
        if hyp_clean is None:
            row["Status"] = "SKIPPED (hypnogram)"
            summary_rows.append(row)
            continue

        orig_df = pd.read_csv(subj["csv_path"], encoding="utf-8-sig")
        orig_df.columns = [c.strip() for c in orig_df.columns]
        orig_epochs = len(orig_df)
        # NS count from original (before expand)
        if "Sleep Stage" in orig_df.columns:
            ns_count = int((orig_df["Sleep Stage"].str.strip().isin(NS_LABELS)).sum())
        elif "StageName" in orig_df.columns:
            ns_count = int((orig_df["StageName"].str.strip().isin(NS_LABELS)).sum())
        else:
            ns_count = 0
        row["Epochs (orig)"] = orig_epochs
        row["Epochs (clean)"] = len(hyp_clean)  # 5-sec expanded
        row["NS removed"] = ns_count

        # Save cleaned CSV for staging model
        clean_csv_path = os.path.join(tmp_dir, f"{sid}.csv")
        hyp_clean.to_csv(clean_csv_path, index=False)

        # Phase 4: EDF → HDF5 preprocessing
        log.info("  Phase 4: EDF → HDF5 conversion")
        raw_hdf5_path = os.path.join(tmp_dir, f"{sid}.hdf5")
        if not preprocess_edf(subj["edf_path"], raw_hdf5_path, log):
            row["Status"] = "ERROR (EDF conversion)"
            summary_rows.append(row)
            continue

        # Log channel count
        with h5py.File(raw_hdf5_path, "r") as hf:
            row["Channels"] = len(hf.keys())

        # Phase 5: Generate embeddings
        log.info("  Phase 5: Generating embeddings")
        if not generate_embeddings(
            raw_hdf5_path, emb_dir, emb_5min_dir, sid,
            emb_model, emb_config, channel_groups,
            device, batch_size, log,
        ):
            row["Status"] = "ERROR (embeddings)"
            summary_rows.append(row)
            continue

        # Delete raw HDF5 — ultra-low RAM
        if os.path.exists(raw_hdf5_path):
            os.remove(raw_hdf5_path)
            log.debug(f"  Deleted temp file: {raw_hdf5_path}")

        emb_path = os.path.join(emb_dir, f"{sid}.hdf5")

        # Phase 6: Sleep staging
        log.info("  Phase 6: Sleep staging inference")
        ss_ok = run_sleep_staging(
            emb_path, clean_csv_path, sid,
            ss_model, ss_config, channel_groups,
            device, output_dir, batch_size, log,
        )

        # Phase 7: Disease prediction
        log.info("  Phase 7: Disease prediction inference")
        dx_ok = run_disease_prediction(
            emb_path, sid, dx_model, dx_config, channel_groups,
            device, data_dir, output_dir, sleepfm_dir,
            batch_size, log,
        )

        # Cleanup temp CSV
        if os.path.exists(clean_csv_path):
            os.remove(clean_csv_path)

        # Finalise status
        elapsed = time.time() - t0
        notes = []
        if not ss_ok:
            notes.append("sleep-staging failed")
        if not dx_ok:
            notes.append("diagnosis failed")
        row["Status"] = "SUCCESS" if (ss_ok and dx_ok) else "PARTIAL"
        row["Notes"] = "; ".join(notes) if notes else f"done in {elapsed:.1f}s"
        summary_rows.append(row)

        # Aggressive cleanup between subjects
        gc.collect()
        torch.cuda.empty_cache()

    # ── Cleanup tmp dir ───────────────────────────────────────────────
    try:
        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # ── Summary Table ─────────────────────────────────────────────────
    log.info("")
    log.info("=" * 80)
    log.info("SUMMARY")
    log.info("=" * 80)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_str = summary_df.to_string(index=False)
        for line in summary_str.split("\n"):
            log.info(line)

        # Save summary CSV
        summary_csv = os.path.join(output_dir, "summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        log.info(f"\nSummary saved → {summary_csv}")

    n_success = sum(1 for r in summary_rows if r["Status"] == "SUCCESS")
    n_partial = sum(1 for r in summary_rows if r["Status"] == "PARTIAL")
    n_skipped = sum(1 for r in summary_rows if "SKIP" in r["Status"])
    n_error   = sum(1 for r in summary_rows if "ERROR" in r["Status"])

    log.info(
        f"\nTotal: {len(summary_rows)} | "
        f"Success: {n_success} | Partial: {n_partial} | "
        f"Skipped: {n_skipped} | Error: {n_error}"
    )
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
