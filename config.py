import numpy as np

# Physical and acoustic parameters
SOUND_SPEED = 343.0
FREQ_RANGE = np.fft.rfftfreq(1024, 1/16000)[1:]
FFT_SIZE = 1024
HOP_LENGTH = 512
SAMPLING_RATE = 16000

# Array geometry parameters
MIC_COUNT = 8
ARRAY_DIAMETER = 0.2

# Optimizer parameters
OPTIMIZER_EPOCHS = 100
OPTIMIZER_POP_SIZE = 50
GA_CROSSOVER_PROB = 0.8
GA_MUTATION_PROB = 0.1
GA_MUTATION_STRENGTH = 0.1
PSO_INERTIA_START = 0.9
PSO_INERTIA_END = 0.4
PSO_C1 = 2.0
PSO_C2 = 2.0

# Acoustic simulation parameters
ROOM_DIMS_RANGE = {'x': [5, 8], 'y': [4, 6], 'z': [2.5, 3.5]}
T60_RANGE = [0.2, 0.9]
SNR_RANGE = [-15, 5]
NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 100
SIM_T60_CONDITIONS = [0.3, 0.6, 0.9]
SIM_SNR_CONDITIONS = [5, 0, -5, -10, -15]

# Dataset paths
LIBRISPEECH_PATH = "./LibriSpeech/dev-other"
NOISEX92_PATH = "./NoiseX-92"
SIM_DATA_DIR = "./simulated_data"
TRAIN_DIR = f"{SIM_DATA_DIR}/train"
TEST_DIR = f"{SIM_DATA_DIR}/test"
PROCESSED_DATA_DIR = "./processed_data"

# Neural network and training parameters
MODEL_TYPE = 'end_to_end' 
MAMBA_D_MODEL = 256
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_N_LAYERS = 4
TRANSFORMER_N_HEAD = 8
TRANSFORMER_N_LAYERS = 4
TRANSFORMER_DIM_FEEDFORWARD = 1024
# --- New LSTM parameters ---
LSTM_HIDDEN_SIZE = 256
LSTM_N_LAYERS = 4
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
EPOCHS = 50
MODEL_SAVE_DIR = "./saved_models"
TRAIN_SEGMENT_SECONDS = 2
NUM_DATALOADER_WORKERS = 0

# Beamforming and evaluation parameters
DOA_SIGNAL = 0
OUTPUT_DIR = "results"
FIG_DPI = 300
VIS_THETA_RANGE = np.linspace(-np.pi, np.pi, 361)
EVAL_CHUNK_SECONDS = 10
