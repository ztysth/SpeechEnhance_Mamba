import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import librosa
import glob
import os
import config

# --- Helper functions ---
def to_stft(signal):
    return librosa.stft(signal, n_fft=config.FFT_SIZE, hop_length=config.HOP_LENGTH)

def compute_time_freq_covariance(stft_data, alpha=0.95):
    M, F, T = stft_data.shape
    R = np.zeros((T, F, M, M), dtype=np.complex128)
    if T == 0: return R
    y_0 = stft_data[:, :, 0].T.reshape(F, M, 1)
    R[0, :, :, :] = y_0 @ y_0.transpose(0, 2, 1).conj()
    for t in range(1, T):
        y_t = stft_data[:, :, t].T.reshape(F, M, 1)
        R[t, :, :, :] = alpha * R[t-1, :, :, :] + (1 - alpha) * (y_t @ y_t.transpose(0, 2, 1).conj())
    return R

# --- The following two dataset classes are only used by preprocess_data.py ---
class AudioCovarianceDataset(Dataset):
    def __init__(self, data_dir, segment_seconds):
        self.mixed_files = sorted(glob.glob(os.path.join(data_dir, "*_mixed.wav")))
        self.noise_files = sorted(glob.glob(os.path.join(data_dir, "*_noise.wav")))
        self.segment_len = int(segment_seconds * config.SAMPLING_RATE)

    def __len__(self): return len(self.mixed_files)
    def __getitem__(self, idx):
        mixed_signal, _ = sf.read(self.mixed_files[idx], dtype='float32')
        noise_signal, _ = sf.read(self.noise_files[idx], dtype='float32')
        num_samples = mixed_signal.shape[0]
        if num_samples >= self.segment_len:
            mixed_segment, noise_segment = mixed_signal[:self.segment_len, :], noise_signal[:self.segment_len, :]
        else:
            pad = ((0, self.segment_len - num_samples), (0, 0))
            mixed_segment, noise_segment = np.pad(mixed_signal, pad), np.pad(noise_signal, pad)
        mixed_stft, noise_stft = to_stft(mixed_segment.T), to_stft(noise_segment.T)
        Rn_target = compute_time_freq_covariance(noise_stft)
        mixed_stft_T = mixed_stft.transpose(2, 1, 0)
        mixed_stft_ri = np.stack([mixed_stft_T.real, mixed_stft_T.imag], axis=-1)
        Rn_target_ri = np.stack([Rn_target.real, Rn_target.imag], axis=-1)
        return torch.from_numpy(mixed_stft_ri).float(), torch.from_numpy(Rn_target_ri).float()

class AudioEnhancementDataset(Dataset):
    def __init__(self, data_dir, segment_seconds):
        self.mixed_files = sorted(glob.glob(os.path.join(data_dir, "*_mixed.wav")))
        self.clean_files = sorted(glob.glob(os.path.join(data_dir, "*_clean.wav")))
        self.segment_len = int(segment_seconds * config.SAMPLING_RATE)
    def __len__(self): return len(self.mixed_files)
    def __getitem__(self, idx):
        mixed_signal, _ = sf.read(self.mixed_files[idx], dtype='float32')
        clean_signal, _ = sf.read(self.clean_files[idx], dtype='float32')
        if clean_signal.ndim == 1: clean_signal = clean_signal[:, np.newaxis]
        num_samples = mixed_signal.shape[0]
        if num_samples >= self.segment_len:
            mixed_segment, clean_segment = mixed_signal[:self.segment_len, :], clean_signal[:self.segment_len, :]
        else:
            pad = ((0, self.segment_len - num_samples), (0, 0))
            mixed_segment, clean_segment = np.pad(mixed_signal, pad), np.pad(clean_signal, pad)
        return torch.from_numpy(mixed_segment.T), torch.from_numpy(clean_segment.T)

# --- This new dataset class will be used by train.py ---
class PrecomputedDataset(Dataset):
    """
    Lightweight dataset that loads preprocessed .pt files directly from disk.
    """
    def __init__(self, cache_dir):
        self.file_paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt cache files found in directory {cache_dir}.\nPlease run 'preprocess_data.py' first to generate cache.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx])

# --- Main creation function now uses PrecomputedDataset ---
def create_dataloaders():
    train_cache_dir = os.path.join(config.PROCESSED_DATA_DIR, config.MODEL_TYPE, 'train')
    test_cache_dir = os.path.join(config.PROCESSED_DATA_DIR, config.MODEL_TYPE, 'test')
    
    train_ds = PrecomputedDataset(train_cache_dir)
    test_ds = PrecomputedDataset(test_cache_dir)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_DATALOADER_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_DATALOADER_WORKERS)
    
    return train_loader, test_loader
