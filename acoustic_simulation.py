import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import glob
from tqdm import tqdm
import config

class AcousticSimulator:
    """Acoustic simulation using PyRoomAcoustics"""

    def __init__(self, mic_positions):
        self.mic_positions = mic_positions
        self.speech_files = glob.glob(os.path.join(config.LIBRISPEECH_PATH, "**/*.flac"), recursive=True)
        self.noise_files = glob.glob(os.path.join(config.NOISEX92_PATH, "*.wav"))
        if not self.speech_files or not self.noise_files:
            raise FileNotFoundError("Please ensure LibriSpeech and NoiseX-92 dataset paths are correct and contain files.")

    def _create_room(self):
        """Create a room with random parameters"""
        room_dim = [
            np.random.uniform(low=config.ROOM_DIMS_RANGE['x'][0], high=config.ROOM_DIMS_RANGE['x'][1]),
            np.random.uniform(low=config.ROOM_DIMS_RANGE['y'][0], high=config.ROOM_DIMS_RANGE['y'][1]),
            np.random.uniform(low=config.ROOM_DIMS_RANGE['z'][0], high=config.ROOM_DIMS_RANGE['z'][1]),
        ]
        t60 = np.random.uniform(low=config.T60_RANGE[0], high=config.T60_RANGE[1])
        
        # Reduce max_order to decrease memory usage and computation time
        e_absorption, max_order = pra.inverse_sabine(t60, room_dim)
        max_order = min(max_order, 10) # Limit maximum reflection order for performance optimization
        
        room = pra.ShoeBox(
            room_dim, fs=config.SAMPLING_RATE, materials=pra.Material(e_absorption), max_order=max_order
        )
        
        center = np.array(room_dim) / 2
        array_center = center + np.random.uniform(-0.5, 0.5, 3)
        
        mic_positions_3d = np.c_[self.mic_positions, np.zeros(config.MIC_COUNT)]
        mic_positions_3d += array_center
        
        room.add_microphone_array(mic_positions_3d.T)
        return room

    def _add_sources(self, room):
        """Add speech and noise sources to the room"""
        speech_path = np.random.choice(self.speech_files)
        speech_signal, _ = sf.read(speech_path)
        
        noise_path = np.random.choice(self.noise_files)
        noise_signal, _ = sf.read(noise_path)

        # Randomly place sound sources
        center_2d = room.shoebox_dim[:2] / 2
        
        angle_speech = np.random.uniform(0, 2 * np.pi)
        radius_speech = np.random.uniform(1, min(center_2d)-0.5) # Ensure not too close to walls
        speech_pos = np.array([center_2d[0] + radius_speech * np.cos(angle_speech), 
                               center_2d[1] + radius_speech * np.sin(angle_speech),
                               np.random.uniform(1, room.shoebox_dim[2]-0.5)]) # Random height

        angle_noise = np.random.uniform(0, 2 * np.pi)
        radius_noise = np.random.uniform(1, min(center_2d)-0.5)
        noise_pos = np.array([center_2d[0] + radius_noise * np.cos(angle_noise),
                              center_2d[1] + radius_noise * np.sin(angle_noise),
                              np.random.uniform(1, room.shoebox_dim[2]-0.5)])
        
        # Ensure speech and noise signals have minimum length to avoid simulation errors
        min_len_samples = config.SAMPLING_RATE * 2 # At least 2 seconds
        if len(speech_signal) < min_len_samples:
             speech_signal = np.pad(speech_signal, (0, min_len_samples - len(speech_signal)))
        if len(noise_signal) < min_len_samples:
             noise_signal = np.pad(noise_signal, (0, min_len_samples - len(noise_signal)))
        
        room.add_source(speech_pos, signal=speech_signal)
        room.add_source(noise_pos, signal=noise_signal)

    def generate_and_save_sample(self, output_path):
        """Generate a simulation sample and save"""
        room = self._create_room()
        self._add_sources(room)

        if len(room.sources) < 2:
            return
            
        # Manually save original signals
        original_speech_signal = room.sources[0].signal.copy()
        original_noise_signal = room.sources[1].signal.copy()

        # Simulate clean speech and clean noise separately
        room.sources[0].signal = original_speech_signal
        room.sources[1].signal = np.zeros_like(original_noise_signal) # Mute noise
        room.simulate()
        assert room.mic_array is not None, "Microphone array not initialized."
        clean_speech_signals = room.mic_array.signals

        room.sources[0].signal = np.zeros_like(original_speech_signal)
        room.sources[1].signal = original_noise_signal
        room.simulate()
        assert room.mic_array is not None, "Microphone array not initialized."
        clean_noise_signals = room.mic_array.signals
        
        # Mix according to SNR
        snr_db = np.random.uniform(*config.SNR_RANGE)
        
        min_len = min(clean_speech_signals.shape[1], clean_noise_signals.shape[1])
        clean_speech_signals = clean_speech_signals[:, :min_len]
        clean_noise_signals = clean_noise_signals[:, :min_len]

        # Manual mixing implementation
        speech_power = np.mean(clean_speech_signals**2)
        noise_power = np.mean(clean_noise_signals**2)
        
        if noise_power > 1e-9:
            # Calculate required noise scaling factor
            required_noise_power = speech_power / (10**(snr_db / 10))
            scale = np.sqrt(required_noise_power / noise_power)
            mixed_signals = clean_speech_signals + clean_noise_signals * scale
        else:
            mixed_signals = clean_speech_signals

        sf.write(f"{output_path}_mixed.wav", mixed_signals.T, config.SAMPLING_RATE)
        sf.write(f"{output_path}_clean.wav", clean_speech_signals.T, config.SAMPLING_RATE)
        sf.write(f"{output_path}_noise.wav", clean_noise_signals.T, config.SAMPLING_RATE)
        
def generate_dataset(mic_positions, data_type, num_samples, start_index=0):
    """
    Generate specified number of dataset samples for training or testing.
    :param mic_positions: Microphone geometry configuration
    :param data_type: 'train' or 'test'
    :param num_samples: Number of samples to generate
    :param start_index: Starting index for filenames
    """
    sim = AcousticSimulator(mic_positions)
    
    if data_type == 'train':
        data_dir = config.TRAIN_DIR
    elif data_type == 'test':
        data_dir = config.TEST_DIR
    else:
        raise ValueError("data_type must be 'train' or 'test'")

    os.makedirs(data_dir, exist_ok=True)
    
    # Use tqdm for progress bar
    end_index = start_index + num_samples
    for i in tqdm(range(start_index, end_index), desc=f"Generating {data_type} samples"):
        sim.generate_and_save_sample(os.path.join(data_dir, f"sample_{i}"))
