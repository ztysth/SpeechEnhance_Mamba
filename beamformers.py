import numpy as np
import torch
import librosa
import webrtcvad
import os
from tqdm import tqdm

import config
from array_utils import steering_vector
from models import MambaEndToEnd, TransformerEndToEnd, LSTMEndToEnd

def _stft(signal_stack):
    """Multi-channel STFT"""
    return librosa.stft(signal_stack, n_fft=config.FFT_SIZE, hop_length=config.HOP_LENGTH)

def _istft(stft_stack, length):
    """Multi-channel iSTFT with length parameter"""
    return librosa.istft(stft_stack, hop_length=config.HOP_LENGTH, length=length)

def delay_and_sum(mic_signals, mic_positions, look_direction_deg):
    """Delay and Sum (DSB) beamformer"""
    X = _stft(mic_signals)
    M, F, T = X.shape
    output_spec = np.zeros((F, T), dtype=X.dtype)
    freq_bins = librosa.fft_frequencies(sr=config.SAMPLING_RATE, n_fft=config.FFT_SIZE)
    look_rad = np.deg2rad(look_direction_deg)
    for f_idx, freq in enumerate(freq_bins):
        a = steering_vector(mic_positions, look_rad, freq)
        w = a.conj() / M
        output_spec[f_idx, :] = np.einsum('i,i...->...', w, X[:, f_idx, :])
    return _istft(output_spec, mic_signals.shape[1])

def traditional_mvdr(mic_signals, mic_positions, look_direction_deg):
    """Traditional MVDR with VAD-based noise covariance estimation"""
    vad = webrtcvad.Vad(3)
    if mic_signals[0].dtype in [np.float32, np.float64]:
        pcm_signal = (mic_signals[0] * 32767).astype(np.int16)
    else:
        pcm_signal = mic_signals[0].astype(np.int16)
        
    frame_len = int(config.SAMPLING_RATE * 30 / 1000)
    X = _stft(mic_signals)
    M, F, T = X.shape
    
    noise_frames_indices = []
    for i in range(0, len(pcm_signal) - frame_len, frame_len):
        try:
            frame = pcm_signal[i:i+frame_len].tobytes()
            if not vad.is_speech(frame, config.SAMPLING_RATE):
                stft_idx = librosa.time_to_frames((i + frame_len / 2) / config.SAMPLING_RATE, 
                                                  sr=config.SAMPLING_RATE, hop_length=config.HOP_LENGTH)
                if stft_idx < T:
                    noise_frames_indices.append(stft_idx)
        except Exception:
            continue

    if not noise_frames_indices: noise_frames_indices = list(range(int(T*0.1)))
    if not noise_frames_indices: return np.zeros(mic_signals.shape[1])
    
    noise_stft = X[:, :, noise_frames_indices]
    noise_stft_transposed = noise_stft.transpose(1, 0, 2) 
    Rn = np.einsum('fmi,fni->fmn', noise_stft_transposed, noise_stft_transposed.conj()) / noise_stft_transposed.shape[2]

    output_spec = np.zeros((F, T), dtype=X.dtype)
    freq_bins = librosa.fft_frequencies(sr=config.SAMPLING_RATE, n_fft=config.FFT_SIZE)
    look_rad = np.deg2rad(look_direction_deg)
    for f_idx, freq in enumerate(freq_bins):
        a = steering_vector(mic_positions, look_rad, freq).reshape(M, 1)
        Rn_f_inv = np.linalg.pinv(Rn[f_idx] + 1e-6 * np.eye(M))
        numerator = Rn_f_inv @ a
        denominator = a.conj().T @ numerator
        if abs(denominator) < 1e-9: continue
        w = numerator / denominator
        output_spec[f_idx, :] = (w.conj().T @ X[:, f_idx, :]).squeeze()
    return _istft(output_spec, mic_signals.shape[1])

def get_model_instance(model_name, device):
    """Load pre-trained model by name"""
    num_freq_bins = config.FFT_SIZE // 2 + 1
    model_parts = model_name.split('_')
    arch = model_parts[0]
    model_type = "_".join(model_parts[1:])
    
    models = {
        ('end_to_end', 'mamba'): MambaEndToEnd,
        ('end_to_end', 'transformer'): TransformerEndToEnd,
        ('end_to_end', 'lstm'): LSTMEndToEnd
    }
    model_class = models.get((model_type, arch.lower()))
    if not model_class: raise ValueError(f"Invalid model name: {model_name}")
    
    model = model_class(num_freq_bins, config.MIC_COUNT)
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found: {model_path}. Please run `train.py` first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def end_to_end_enhancer(mic_signals, model_name, device):
    """End-to-end model inference: NN predicts mask and applies it"""
    model = get_model_instance(model_name, device)
    
    mixed_wav_t = torch.from_numpy(mic_signals).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        enhanced_wav = model(mixed_wav_t)

    return enhanced_wav.squeeze(0).cpu().numpy()
