import numpy as np
from scipy.linalg import inv
import config

def steering_vector(mic_positions, theta, freq):
    """Calculate steering vector for given frequency and direction"""
    # Avoid division by zero
    if freq < 1e-6:
        return np.ones(mic_positions.shape[0], dtype=np.complex128)
        
    k = 2 * np.pi * freq / config.SOUND_SPEED
    # mic_positions: (M, 2), theta: scalar
    # x*cos(theta) + y*sin(theta)
    tau = mic_positions @ np.array([np.cos(theta), np.sin(theta)])
    return np.exp(1j * k * tau)

def calculate_diffuse_noise_cov(mic_positions, freq):
    """
    Calculate spatial covariance matrix for ideal diffuse (isotropic) noise field
    Matrix element (i, j) is sinc(2 * pi * f * d_ij / c)
    """
    m = mic_positions.shape[0]
    if freq < 1e-6:
        return np.eye(m)
        
    # Calculate distance matrix between microphones
    dist_matrix = np.sqrt(np.sum((mic_positions[:, np.newaxis, :] - mic_positions[np.newaxis, :, :])**2, axis=-1))
    
    k = 2 * np.pi * freq / config.SOUND_SPEED
    # Use np.sinc(x) = sin(pi*x)/(pi*x)
    # sinc(k*d) = sinc(2*pi*f*d/c)
    cov_matrix = np.sinc(dist_matrix * k / np.pi)
    return cov_matrix

class ArrayEvaluator:
    """
    Array performance evaluator
    Based on robust superdirective beamforming principle, calculates WNG and DI.
    """
    def __init__(self, mic_positions):
        self.mic_positions = np.array(mic_positions)
        if self.mic_positions.ndim != 2 or self.mic_positions.shape[1] != 2:
            raise ValueError("Microphone positions must be a (M, 2) array")
        self.M = self.mic_positions.shape[0]

    def _calculate_optimal_weights(self, look_direction, freq, diagonal_loading_param=1e-5):
        """
        Calculate robust weights that maximize directivity index (Robust Superdirective)
        """
        a_look = steering_vector(self.mic_positions, look_direction, freq)
        
        R_n = calculate_diffuse_noise_cov(self.mic_positions, freq)
        
        try:
            # Use pseudo-inverse for numerical stability
            R_n_inv = np.linalg.pinv(R_n + diagonal_loading_param * np.eye(self.M))
        except np.linalg.LinAlgError:
             return (a_look.conj() / self.M)
        
        numerator = R_n_inv @ a_look
        denominator = np.vdot(a_look, numerator)
        
        if np.abs(denominator) < 1e-9:
            return (a_look.conj() / self.M)

        return numerator / denominator

    def beampattern(self, look_direction, freq, theta_range):
        """Calculate beam pattern at specified frequency and direction"""
        w = self._calculate_optimal_weights(look_direction, freq)
        patterns = []
        for theta in theta_range:
            a = steering_vector(self.mic_positions, theta, freq)
            patterns.append(np.abs(np.vdot(w, a)))
        return np.array(patterns)

    def wng(self, w):
        """Calculate white noise gain (WNG) for given weights"""
        # WNG defined as 1 / (w^H * w)
        norm_w_sq = np.vdot(w, w)
        if norm_w_sq.real < 1e-9:
            return 0
        return 1 / norm_w_sq.real

    def di(self, w, look_direction, freq):
        """Calculate directivity index (DI) for given weights"""
        # DI defined as |w^H * a_s|^2 / (w^H * R_n * w)
        # Since weights are normalized such that |w^H * a_s| = 1, DI = 1 / (w^H * R_n * w)
        R_n = calculate_diffuse_noise_cov(self.mic_positions, freq)
        denominator = np.vdot(w, R_n @ w)
        if denominator.real < 1e-9:
            return 0
        directivity_factor = 1 / denominator.real
        return 10 * np.log10(directivity_factor)

    def evaluate_broadband(self, look_direction=0):
        """Calculate average performance across the frequency band"""
        avg_wng_db = 0
        avg_di = 0
        
        look_rad = np.deg2rad(look_direction)

        valid_freqs = [f for f in config.FREQ_RANGE if f > 1.0]
        if not valid_freqs:
            return -100, 0

        for freq in valid_freqs:
            w = self._calculate_optimal_weights(look_rad, freq)
            
            wng_val = self.wng(w)
            if wng_val > 1e-9:
                avg_wng_db += 10 * np.log10(wng_val)
            
            di_val = self.di(w, look_rad, freq)
            avg_di += di_val

        num_freqs = len(valid_freqs)
        return avg_wng_db / num_freqs, avg_di / num_freqs
    
    @staticmethod
    def get_fitness_function():
        """Fitness function interface for optimizers"""
        def fitness(positions):
            evaluator = ArrayEvaluator(positions)
            avg_wng, avg_di = evaluator.evaluate_broadband()
            
            di_min = 5.0
            penalty = max(0, (di_min - avg_di)) * 10
            
            return avg_wng - penalty
        return fitness
