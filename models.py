import torch
import torch.nn as nn
from mamba_ssm import Mamba
import config

# STFT helper module
class STFT:
    """A helper class that encapsulates STFT and iSTFT operations to ensure parameter consistency"""
    def __init__(self, n_fft, hop_length, win_length, window):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window

    def forward(self, wav):
        """Perform STFT"""
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(wav.device),
            return_complex=True
        )

    def inverse(self, stft, length):
        """Perform iSTFT"""
        wav = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(stft.device),
            length=length
        )
        return wav

# End-to-end model (predicts mask)
class EnhancementFormer(nn.Module):
    """
    End-to-end model base class: takes multi-channel WAV and predicts a time-frequency mask.
    """
    def __init__(self, num_freq_bins, num_channels, feature_dim):
        super().__init__()
        self.num_channels = num_channels
        self.num_freq_bins = num_freq_bins
        
        self.stft = STFT(
            n_fft=config.FFT_SIZE,
            hop_length=config.HOP_LENGTH,
            win_length=config.FFT_SIZE,
            window=torch.hann_window(config.FFT_SIZE)
        )
        
        self.input_fc = nn.Linear(num_channels * num_freq_bins * 2, feature_dim)
        self.ln1 = nn.LayerNorm(feature_dim)
        self.activation = nn.GELU()
        
        self.sequence_model = None
        
        self.ln2 = nn.LayerNorm(feature_dim)
        self.output_fc = nn.Linear(feature_dim, num_freq_bins * 2)

    def forward(self, x_wav):
        B, M, L = x_wav.shape
        
        mixed_stft = self.stft.forward(x_wav.reshape(B * M, L))
        _, F, T = mixed_stft.shape
        mixed_stft = mixed_stft.reshape(B, M, F, T)
        
        x = mixed_stft.permute(0, 3, 1, 2) 
        x_ri = torch.view_as_real(x.contiguous()).flatten(start_dim=2)
        
        features = self.input_fc(x_ri)
        features = self.ln1(features)
        features = self.activation(features)
        
        if self.sequence_model:
            features = self.sequence_model(features)
            if isinstance(self.sequence_model, nn.LSTM): # LSTM output is tuple (output, (h_n, c_n))
                features = features[0]
            features = torch.tanh(features)
            
        features = self.ln2(features)
        mask_ri = self.output_fc(features).reshape(B, T, F, 2)
        
        predicted_mask = torch.view_as_complex(mask_ri.contiguous())
        predicted_mask = predicted_mask.permute(0, 2, 1)
        
        # Apply mask to all channels
        enhanced_stft_multichannel = mixed_stft * predicted_mask.unsqueeze(1)

        # Return enhanced time-domain signal from reference channel (first microphone)
        enhanced_stft_ref = enhanced_stft_multichannel[:, 0, :, :]
        enhanced_wav = self.stft.inverse(enhanced_stft_ref, L)
        
        return enhanced_wav

class MambaEndToEnd(EnhancementFormer):
    def __init__(self, num_freq_bins, num_channels):
        super().__init__(num_freq_bins, num_channels, config.MAMBA_D_MODEL)
        self.sequence_model = nn.Sequential(
            *[Mamba(d_model=config.MAMBA_D_MODEL, d_state=config.MAMBA_D_STATE, d_conv=config.MAMBA_D_CONV) 
              for _ in range(config.MAMBA_N_LAYERS)]
        )

class TransformerEndToEnd(EnhancementFormer):
    def __init__(self, num_freq_bins, num_channels):
        super().__init__(num_freq_bins, num_channels, config.MAMBA_D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.MAMBA_D_MODEL,
            nhead=config.TRANSFORMER_N_HEAD,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            batch_first=True
        )
        self.sequence_model = nn.TransformerEncoder(encoder_layer, num_layers=config.TRANSFORMER_N_LAYERS)

# --- New LSTM model ---
class LSTMEndToEnd(EnhancementFormer):
    def __init__(self, num_freq_bins, num_channels):
        super().__init__(num_freq_bins, num_channels, config.LSTM_HIDDEN_SIZE)
        self.sequence_model = nn.LSTM(
            input_size=config.LSTM_HIDDEN_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_N_LAYERS,
            batch_first=True,
            bidirectional=False # Unidirectional LSTM
        )
