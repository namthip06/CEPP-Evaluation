#!/usr/bin/env python3
"""
Sleep Staging Prediction Script

This script predicts sleep stages (Wake, Stage 1, Stage 2, Stage 3, REM) using
the pretrained SleepFM sleep staging model and the generated embeddings.

It processes subjects in /home/nummm/Documents/CEPP/rawEEG/ one at a time
to avoid RAM issues.

Each subject should have:
- Embeddings: /custom/embeddings/{subject_id}_embeddings.hdf5
- Ground truth labels: /home/nummm/Documents/CEPP/rawEEG/{subject_id}/{subject_id}.csv

The script will:
1. Load the pretrained sleep staging model
2. Load embeddings for each subject
3. Load ground truth labels
4. Generate sleep stage predictions
5. Save predictions to /custom/predictions/{subject_id}_predictions.csv
"""

import os
import sys
import torch
import torch.nn as nn
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import traceback
from tqdm import tqdm

# Add parent directory to path
# sys.path.append("..")
# sys.path.append("../sleepfm")
# from sleepfm.utils import load_config, count_parameters
# from sleepfm.models.models import SleepEventLSTMClassifier

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from sleepfm.utils import load_config, count_parameters
from sleepfm.models.models import SleepEventLSTMClassifier

class SleepStagingPredictor:
    """Predict sleep stages using pretrained SleepFM model."""
    
    def __init__(self, model_path, device="cuda"):
        """
        Initialize the sleep staging predictor.
        
        Args:
            model_path: Path to the pretrained model checkpoint
            device: Device to use for inference (cuda or cpu)
        """
        self.model_path = model_path
        # self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        self.config = load_config(config_path)
        
        # Model parameters
        self.model_params = self.config["model_params"]
        self.num_classes = self.model_params["num_classes"]
        self.max_channels = self.config.get("max_channels", 4)
        
        # Sleep stage mapping
        self.stage_names = {
            0: "Wake",
            1: "Stage 1",
            2: "Stage 2",
            3: "Stage 3",
            4: "REM"
        }
        
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of classes: {self.num_classes}")
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the pretrained model."""
        logger.info("Loading pretrained sleep staging model...")
        
        # Initialize model
        self.model = SleepEventLSTMClassifier(**self.model_params)
        
        # Load checkpoint
        checkpoint_path = os.path.join(self.model_path, "best.pth")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Handle DataParallel state dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        total_layers, total_params = count_parameters(self.model)
        logger.info(f"Model loaded: {total_params / 1e6:.2f}M parameters, {total_layers} layers")
    
    def load_embeddings(self, embedding_path):
        """
        Load embeddings from HDF5 file.
        
        Args:
            embedding_path: Path to the embedding HDF5 file
            
        Returns:
            Embeddings as numpy array with shape [channels, sequence, embed_dim]
        """
        embeddings = []
        with h5py.File(embedding_path, 'r') as hf:
            # Load BAS modality (EEG channels)
            if 'BAS' in hf:
                embeddings.append(hf['BAS'][:])
            
            # Optionally load other modalities if available
            for modality in ['RESP', 'EKG', 'EMG']:
                if modality in hf:
                    embeddings.append(hf[modality][:])
        
        if not embeddings:
            raise ValueError(f"No embeddings found in {embedding_path}")
        
        # Stack embeddings: [channels, sequence, embed_dim]
        embeddings = np.array(embeddings)
        
        # Remove the extra dimension if present (from unsqueeze in embedding generation)
        if embeddings.ndim == 4 and embeddings.shape[2] == 1:
            embeddings = embeddings.squeeze(2)
        
        return embeddings
    
    def load_labels(self, label_path):
        """
        Load ground truth labels from CSV file.
        
        Args:
            label_path: Path to the label CSV file
            
        Returns:
            Labels as numpy array
        """
        df = pd.read_csv(label_path)
        
        # Handle different column names
        if 'StageNumber' in df.columns:
            labels = df['StageNumber'].to_numpy()
        elif 'stage' in df.columns:
            labels = df['stage'].to_numpy()
        else:
            raise ValueError(f"Cannot find stage labels in {label_path}")
        
        # Replace -1 with 0 (Wake)
        labels = np.where(labels == -1, 0, labels)
        
        return labels
    
    def predict_single_subject(self, embedding_path, label_path, output_path):
        """
        Predict sleep stages for a single subject.
        
        Args:
            embedding_path: Path to the embedding HDF5 file
            label_path: Path to the ground truth label CSV file
            output_path: Path to save predictions CSV
            
        Returns:
            True if successful, False otherwise
        """
        subject_id = os.path.basename(embedding_path).replace('_embeddings.hdf5', '')
        logger.info(f"Processing: {subject_id}")
        
        try:
            # Load embeddings and labels
            embeddings = self.load_embeddings(embedding_path)
            labels = self.load_labels(label_path)
            
            logger.info(f"  Embeddings shape: {embeddings.shape}")
            logger.info(f"  Labels shape: {labels.shape}")
            
            # Prepare input tensor
            # embeddings shape: [channels, sequence, embed_dim]
            # model expects: [batch, channels, sequence, embed_dim]
            x_data = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
            
            # Ensure we have the right number of channels
            num_channels = x_data.shape[1]
            if num_channels < self.max_channels:
                # Pad with zeros if we have fewer channels
                padding = torch.zeros(1, self.max_channels - num_channels, 
                                     x_data.shape[2], x_data.shape[3])
                x_data = torch.cat([x_data, padding], dim=1)
            elif num_channels > self.max_channels:
                # Take only the first max_channels
                x_data = x_data[:, :self.max_channels, :, :]
            
            # Create padding mask (0 for valid, 1 for padding)
            # Shape: [batch, channels, sequence]
            padded_matrix = torch.zeros(1, self.max_channels, x_data.shape[2])
            if num_channels < self.max_channels:
                padded_matrix[:, num_channels:, :] = 1
            
            # Move to device
            x_data = x_data.to(self.device)
            padded_matrix = padded_matrix.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs, mask = self.model(x_data, padded_matrix)
                # outputs shape: [batch, sequence, num_classes]
                
                # Get probabilities
                probabilities = torch.softmax(outputs, dim=-1)
                
                # Get predicted classes
                predictions = torch.argmax(probabilities, dim=-1)
                
                # Move to CPU and convert to numpy
                predictions = predictions.cpu().numpy()[0]  # Remove batch dimension
                probabilities = probabilities.cpu().numpy()[0]  # Remove batch dimension
            
            # Ensure predictions and labels have the same length
            min_length = min(len(predictions), len(labels))
            predictions = predictions[:min_length]
            probabilities = probabilities[:min_length]
            labels = labels[:min_length]
            
            # Create output dataframe
            output_data = {
                'Epoch': np.arange(min_length),
                'GroundTruth': labels,
                'Predicted': predictions,
                'Confidence_Wake': probabilities[:, 0],
                'Confidence_Stage1': probabilities[:, 1],
                'Confidence_Stage2': probabilities[:, 2],
                'Confidence_Stage3': probabilities[:, 3],
                'Confidence_REM': probabilities[:, 4]
            }
            
            output_df = pd.DataFrame(output_data)
            
            # Save to CSV
            output_df.to_csv(output_path, index=False)
            
            # Calculate accuracy
            accuracy = (predictions == labels).mean() * 100
            logger.info(f"  ✓ Predictions saved: {os.path.basename(output_path)}")
            logger.info(f"  Accuracy: {accuracy:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {subject_id}: {str(e)}")
            traceback.print_exc()
            return False


def main():
    """Main function to process all subjects."""

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    print("base_dir: ", base_dir)
    
    # Configuration
    raw_eeg_dir = "/home/nummm/Documents/CEPP/rawEEG"
    embeddings_dir = os.path.join(base_dir, "custom/embeddings")
    model_path = os.path.join(base_dir, "sleepfm/checkpoints/model_sleep_staging")
    output_dir = os.path.join(base_dir, "custom/predictions")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Sleep Staging Prediction")
    logger.info("="*60)
    logger.info(f"Raw EEG directory: {raw_eeg_dir}")
    logger.info(f"Embeddings directory: {embeddings_dir}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    # Initialize predictor
    predictor = SleepStagingPredictor(model_path)
    
    # Get all embedding files
    if not os.path.exists(embeddings_dir):
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        logger.error("Please run generate_embeddings.py first")
        return
    
    embedding_files = sorted([
        f for f in os.listdir(embeddings_dir)
        if f.endswith('_embeddings.hdf5')
    ])
    
    logger.info(f"\nFound {len(embedding_files)} embedding files\n")
    
    # Process each subject
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for i, embedding_file in enumerate(embedding_files, 1):
        subject_id = embedding_file.replace('_embeddings.hdf5', '')
        embedding_path = os.path.join(embeddings_dir, embedding_file)
        label_path = os.path.join(raw_eeg_dir, subject_id, f"{subject_id}.csv")
        output_path = os.path.join(output_dir, f"{subject_id}_predictions.csv")
        
        logger.info(f"\n[{i}/{len(embedding_files)}] {subject_id}")
        
        # Check if label file exists
        if not os.path.exists(label_path):
            logger.warning(f"  ⚠ Skipping: Label file not found")
            skip_count += 1
            continue
        
        # Process the subject
        result = predictor.predict_single_subject(embedding_path, label_path, output_path)
        
        if result:
            success_count += 1
        else:
            error_count += 1
        
        # Clear CUDA cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total subjects: {len(embedding_files)}")
    logger.info(f"✅ Successfully processed: {success_count}")
    logger.info(f"⚠  Skipped (no labels): {skip_count}")
    logger.info(f"❌ Errors: {error_count}")
    logger.info("="*60)
    logger.info(f"\nPredictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
