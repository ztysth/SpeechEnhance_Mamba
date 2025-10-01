import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np

import config
from models import MambaEndToEnd, TransformerEndToEnd, LSTMEndToEnd
from data_loader import PrecomputedDataset

# End-to-end model SI-SNR loss function
class SISNRLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio loss function"""
    def __init__(self, epsilon=1e-8):
        super(SISNRLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, estimate, target):
        if estimate.dim() == 3: estimate = estimate.squeeze(1)
        if target.dim() == 3: target = target.squeeze(1)
        estimate = estimate - torch.mean(estimate, dim=1, keepdim=True)
        target = target - torch.mean(target, dim=1, keepdim=True)
        s_target = torch.sum(estimate * target, dim=1, keepdim=True) * target / (torch.sum(target * target, dim=1, keepdim=True) + self.epsilon)
        e_noise = estimate - s_target
        si_snr = 10 * torch.log10(torch.sum(s_target * s_target, dim=1) / (torch.sum(e_noise * e_noise, dim=1) + self.epsilon) + self.epsilon)
        return -torch.mean(si_snr)

# Training and validation loop
def run_epoch(model, loader, criterion, optimizer, device, is_training, current_epoch):
    """Run one training or validation epoch"""
    model.train() if is_training else model.eval()
    total_loss = 0
    desc = "Training" if is_training else "Validation"
    pbar = tqdm(loader, desc=f"Epoch {current_epoch+1}/{config.EPOCHS} [{desc}]")

    for data in pbar:
        if is_training: optimizer.zero_grad()

        if config.MODEL_TYPE == 'end_to_end':
            mixed_wav, clean_wav = data
            mixed_wav, clean_wav = mixed_wav.to(device), clean_wav.to(device)
            enhanced_wav = model(mixed_wav)
            loss = criterion(enhanced_wav, clean_wav[:, 0, :])

        if torch.isnan(loss):
            return float('nan')

        if is_training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / len(loader)

# Training function
def train(model_type, arch):
    """Complete model training pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header(f"Starting training: type={model_type}, architecture={arch}")
    print(f"Using device: {device}")

    config.MODEL_TYPE = model_type
    num_freq_bins = config.FFT_SIZE // 2 + 1
    
    models = {
        ('end_to_end', 'mamba'): MambaEndToEnd,
        ('end_to_end', 'transformer'): TransformerEndToEnd,
        ('end_to_end', 'lstm'): LSTMEndToEnd
    }
    model_class = models.get((model_type, arch))
    if not model_class: raise ValueError("Invalid model type or architecture")
    model = model_class(num_freq_bins, config.MIC_COUNT).to(device)

    train_cache_dir = os.path.join(config.PROCESSED_DATA_DIR, model_type, 'train')
    test_cache_dir = os.path.join(config.PROCESSED_DATA_DIR, model_type, 'test')
    if not os.path.exists(train_cache_dir) or not os.listdir(train_cache_dir):
        print(f"Error: Preprocessed data '{train_cache_dir}' not found. Please run 'preprocess_data.py' for '{model_type}' first.")
        return

    train_ds = PrecomputedDataset(train_cache_dir)
    test_ds = PrecomputedDataset(test_cache_dir)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_DATALOADER_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_DATALOADER_WORKERS)
    
    criterion = SISNRLoss().to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=False)

    best_loss = float('inf')
    model_save_name = f"{arch}_{model_type}_best.pth"
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    for epoch in range(config.EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, True, epoch)
        if np.isnan(train_loss): break
        with torch.no_grad():
            val_loss = run_epoch(model, test_loader, criterion, None, device, False, epoch)
        if np.isnan(val_loss): break
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train loss = {train_loss:.6f}, Val loss = {val_loss:.6f}, LR = {current_lr:.1e}")
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, model_save_name))

def print_header(title):
    """Print bordered title"""
    print("\n" + "="*70)
    print(f"| {title.center(66)} |")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train neural network models")
    parser.add_argument('--model_type', type=str, required=True, choices=['end_to_end'], help="Select model paradigm")
    parser.add_argument('--arch', type=str, required=True, choices=['mamba', 'transformer', 'lstm'], help="Select neural network architecture")
    args = parser.parse_args()
    
    train(model_type=args.model_type, arch=args.arch)
