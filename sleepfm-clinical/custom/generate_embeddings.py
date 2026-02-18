#!/usr/bin/env python3
"""
Embedding Generation Script for Raw EEG Data

This script extracts sleep representations using the pretrained SleepFM model.
It processes folders in /home/nummm/Documents/CEPP/rawEEG/ one at a time
to avoid RAM issues.

Each folder should contain:
- {folder_name}.hdf5 (created by preprocess.py)
- {folder_name}.csv (sleep stage labels)

The script will:
1. Load the pretrained SleepFM model (model_base)
2. Extract embeddings from EEG channels only (BAS modality)
3. Save embeddings to /custom folder
4. Process folders sequentially (no batching/parallel processing)
"""

import os
import sys
import torch
import h5py
import numpy as np
from pathlib import Path
from loguru import logger
import traceback
from tqdm import tqdm

# Add parent directory to path
# sys.path.append("..")
# sys.path.append("../sleepfm")
# from utils import load_config, load_data, count_parameters
# from models.dataset import SetTransformerDataset, collate_fn
# from models.models import SetTransformer

# หาทางกลับไปยัง Root Directory (zou-group-sleepfm-clinical)
# __file__ คือ path ของไฟล์ปัจจุบันใน custom/
# os.path.dirname(__file__) คือ path ของโฟลเดอร์ custom/
# '..' คือถอยหลังออกไป 1 ก้าวสู่ Root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

print(root_path)

from sleepfm.utils import load_config, load_data, count_parameters
from sleepfm.models.dataset import SetTransformerDataset, collate_fn
from sleepfm.models.models import SetTransformer

class EmbeddingGenerator:
    """Generate embeddings from HDF5 files using pretrained SleepFM model."""
    
    def __init__(self, model_path, channel_groups_path, device="cuda"):
        """
        Initialize the embedding generator.
        
        Args:
            model_path: Path to the pretrained model checkpoint
            channel_groups_path: Path to channel groups configuration
            device: Device to use for inference (cuda or cpu)
        """
        self.model_path = model_path
        # self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        print("Device:", self.device)
        
        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        self.config = load_config(config_path)
        self.channel_groups = load_data(channel_groups_path)
        
        # Model parameters
        self.modality_types = self.config["modality_types"]
        self.in_channels = self.config["in_channels"]
        self.patch_size = self.config["patch_size"]
        self.embed_dim = self.config["embed_dim"]
        self.num_heads = self.config["num_heads"]
        self.num_layers = self.config["num_layers"]
        self.pooling_head = self.config["pooling_head"]
        self.dropout = 0.0  # No dropout during inference
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Modality types: {self.modality_types}")
        logger.info(f"Embed dim: {self.embed_dim}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the pretrained model."""
        logger.info("Loading pretrained model...")
        
        # Initialize model
        self.model = SetTransformer(
            self.in_channels,
            self.patch_size,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            pooling_head=self.pooling_head,
            dropout=self.dropout
        )
        
        # Load checkpoint
        checkpoint_path = os.path.join(self.model_path, "best.pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DataParallel state dict
        state_dict = checkpoint["state_dict"]
        if list(state_dict.keys())[0].startswith('module.'):
            # Remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        total_layers, total_params = count_parameters(self.model)
        logger.info(f"Model loaded: {total_params / 1e6:.2f}M parameters, {total_layers} layers")
    
    def process_single_file(self, hdf5_path, output_dir):
        """
        Process a single HDF5 file and generate embeddings.
        
        Args:
            hdf5_path: Path to the HDF5 file
            output_dir: Directory to save embeddings
            
        Returns:
            True if successful, False otherwise
        """
        subject_id = os.path.basename(hdf5_path).replace('.hdf5', '')
        logger.info(f"Processing: {subject_id}")
        
        try:
            # Create dataset for single file
            dataset = SetTransformerDataset(
                self.config,
                self.channel_groups,
                hdf5_paths=[hdf5_path],
                split="test"
            )
            
            # Create dataloader with batch_size=1 to process one chunk at a time
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=0,  # No parallel workers to save RAM
                shuffle=False,
                collate_fn=collate_fn
            )
            
            logger.info(f"  Total chunks: {len(dataloader)}")
            
            # Prepare output paths
            output_path = os.path.join(output_dir, f"{subject_id}_embeddings.hdf5")
            output_5min_path = os.path.join(output_dir, f"{subject_id}_embeddings_5min.hdf5")
            
            # Process chunks
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"  Chunks", leave=False)):
                    batch_data, mask_list, file_paths, dset_names_list, chunk_starts = batch
                    (bas, resp, ekg, emg) = batch_data
                    (mask_bas, mask_resp, mask_ekg, mask_emg) = mask_list
                    
                    # Move to device - only process BAS (EEG) modality
                    bas = bas.to(self.device, dtype=torch.float)
                    mask_bas = mask_bas.to(self.device, dtype=torch.bool)
                    
                    # Generate embeddings for BAS modality only
                    embedding = self.model(bas, mask_bas)
                    
                    # embedding is a tuple: (5min_aggregated, per_epoch)
                    embedding_5min = embedding[0].unsqueeze(1)  # [batch, 1, embed_dim]
                    embedding_epoch = embedding[1]  # [batch, num_epochs, embed_dim]
                    
                    # Save embeddings
                    chunk_start = chunk_starts[0]
                    
                    # Save 5-minute aggregated embeddings
                    with h5py.File(output_5min_path, 'a') as hdf5_file:
                        modality_type = "BAS"
                        if modality_type in hdf5_file:
                            dset = hdf5_file[modality_type]
                            chunk_start_correct = chunk_start // (self.embed_dim * 5 * 60)
                            chunk_end = chunk_start_correct + embedding_5min[0].shape[0]
                            if dset.shape[0] < chunk_end:
                                dset.resize((chunk_end,) + embedding_5min[0].shape[1:])
                            dset[chunk_start_correct:chunk_end] = embedding_5min[0].cpu().numpy()
                        else:
                            hdf5_file.create_dataset(
                                modality_type,
                                data=embedding_5min[0].cpu().numpy(),
                                chunks=(self.embed_dim,) + embedding_5min[0].shape[1:],
                                maxshape=(None,) + embedding_5min[0].shape[1:]
                            )
                    
                    # Save per-epoch embeddings
                    with h5py.File(output_path, 'a') as hdf5_file:
                        modality_type = "BAS"
                        if modality_type in hdf5_file:
                            dset = hdf5_file[modality_type]
                            chunk_start_correct = chunk_start // (self.embed_dim * 5)
                            chunk_end = chunk_start_correct + embedding_epoch[0].shape[0]
                            if dset.shape[0] < chunk_end:
                                dset.resize((chunk_end,) + embedding_epoch[0].shape[1:])
                            dset[chunk_start_correct:chunk_end] = embedding_epoch[0].cpu().numpy()
                        else:
                            hdf5_file.create_dataset(
                                modality_type,
                                data=embedding_epoch[0].cpu().numpy(),
                                chunks=(self.embed_dim,) + embedding_epoch[0].shape[1:],
                                maxshape=(None,) + embedding_epoch[0].shape[1:]
                            )
            
            logger.info(f"  ✓ Embeddings saved:")
            logger.info(f"    - {os.path.basename(output_path)}")
            logger.info(f"    - {os.path.basename(output_5min_path)}")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {subject_id}: {str(e)}")
            traceback.print_exc()
            return False


def main():
    """Main function to process all folders."""

    # 1. หา Root Path ของโปรเจกต์ (ขึ้นไป 1 ระดับจาก custom/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Configuration
    raw_eeg_dir = "/home/nummm/Documents/CEPP/rawEEG"
    model_path = os.path.join(base_dir, "sleepfm/checkpoints/model_base")
    channel_groups_path = os.path.join(base_dir, "sleepfm/configs/channel_groups.json")
    output_dir = os.path.join(base_dir, "custom/embeddings")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("SleepFM Embedding Generation")
    logger.info("="*60)
    logger.info(f"Raw EEG directory: {raw_eeg_dir}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    # Initialize generator
    generator = EmbeddingGenerator(model_path, channel_groups_path)
    
    # Get all folders
    if not os.path.exists(raw_eeg_dir):
        logger.error(f"Directory not found: {raw_eeg_dir}")
        return
    
    folders = sorted([
        os.path.join(raw_eeg_dir, d)
        for d in os.listdir(raw_eeg_dir)
        if os.path.isdir(os.path.join(raw_eeg_dir, d))
    ])
    
    logger.info(f"\nFound {len(folders)} folders to process\n")
    
    # Process each folder
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, folder_path in enumerate(folders, 1):
        folder_name = os.path.basename(folder_path)
        hdf5_path = os.path.join(folder_path, f"{folder_name}.hdf5")
        
        logger.info(f"\n[{i}/{len(folders)}] {folder_name}")
        
        # Check if HDF5 file exists
        if not os.path.exists(hdf5_path):
            logger.warning(f"  ⚠ Skipping: HDF5 file not found")
            skip_count += 1
            continue
        
        # Process the file
        result = generator.process_single_file(hdf5_path, output_dir)
        
        if result:
            success_count += 1
        else:
            error_count += 1
        
        # Clear CUDA cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total folders: {len(folders)}")
    logger.info(f"✅ Successfully processed: {success_count}")
    logger.info(f"⚠  Skipped (no HDF5): {skip_count}")
    logger.info(f"❌ Errors: {error_count}")
    logger.info("="*60)
    logger.info(f"\nEmbeddings saved to: {output_dir}")


if __name__ == "__main__":
    main()
