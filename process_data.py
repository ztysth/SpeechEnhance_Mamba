import torch
import os
import argparse
from tqdm import tqdm
import gc
import math
import numpy as np
from torch.utils.data import DataLoader, default_collate
import librosa

import config
from data_loader import AudioCovarianceDataset, AudioEnhancementDataset

def custom_collate_fn(batch):
    """
    Custom collate_fn to handle potentially None samples.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)

def preprocess_batch(batch, target_dir, start_idx):
    """
    Process and save a batch of data
    """
    if batch is None:
        return 0

    inputs_batch, targets_batch = batch
    
    num_in_batch = inputs_batch.size(0)
    for i in range(num_in_batch):
        sample_tuple = (inputs_batch[i].clone(), targets_batch[i].clone())
        try:
            torch.save(sample_tuple, os.path.join(target_dir, f"sample_{start_idx + i}.pt"))
        except Exception as e:
            continue
    return num_in_batch

def estimate_memory_usage(model_type, sample_wav_path):
    """Estimate memory usage per sample"""
    try:
        y, sr = librosa.load(sample_wav_path, sr=config.SAMPLING_RATE)
        segment_len = int(config.TRAIN_SEGMENT_SECONDS * config.SAMPLING_RATE)
        y = y[:segment_len]

        mem_per_sample = y.nbytes * config.MIC_COUNT * 2 # Mixed signal + clean signal
        return mem_per_sample
    except Exception as e:
        return 50 * 1024 * 1024 # Return conservative default value 50MB

def preprocess_data_pipeline(model_type):
    """Preprocess and cache data for specified model type (memory optimized version)"""
    print_header(f"Preprocessing/Caching data for '{model_type}' model")
    config.MODEL_TYPE = model_type
    
    import psutil
    available_memory = psutil.virtual_memory().available * 0.7
    
    for data_split in ['train', 'test']:
        print(f"\nProcessing {data_split} data...")
        source_dir = getattr(config, f"{data_split.upper()}_DIR")
        target_dir = os.path.join(config.PROCESSED_DATA_DIR, model_type, data_split)
        os.makedirs(target_dir, exist_ok=True)
        
        wav_files = [f for f in os.listdir(source_dir) if f.endswith('_mixed.wav')]
        if not wav_files:
            print(f"Warning: No WAV files found in {source_dir}. Skipping...")
            continue
            
        sample_wav_path = os.path.join(source_dir, wav_files[0])
        mem_per_sample = estimate_memory_usage(model_type, sample_wav_path)
        batch_size = max(1, min(32, int(available_memory / (mem_per_sample * 2))))
        
        print(f"Automatically selected batch size based on available memory: {batch_size}")

        DatasetClass = AudioEnhancementDataset
        raw_dataset = DatasetClass(source_dir, segment_seconds=config.TRAIN_SEGMENT_SECONDS)
        
        data_loader = DataLoader(
            raw_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.NUM_DATALOADER_WORKERS,
            pin_memory=False,
            collate_fn=custom_collate_fn
        )
        
        processed_count = 0
        pbar = tqdm(data_loader, desc=f"Processing {data_split} data")
        for batch in pbar:
            try:
                num_processed = preprocess_batch(batch, target_dir, processed_count)
                processed_count += num_processed
                
                if processed_count % (batch_size * 5) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                continue
                
    print(f"\n'{model_type}' model data preprocessing completed!")

def print_header(title):
    print("\n" + "="*70)
    print(f"| {title.center(66)} |")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess and cache training data")
    parser.add_argument('--model_type', type=str, required=True, choices=['end_to_end'], help="Select model type to cache data for (directory name)")
    args = parser.parse_args()
    
    preprocess_data_pipeline(args.model_type)
