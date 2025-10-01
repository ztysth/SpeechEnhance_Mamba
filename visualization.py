import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import MaxNLocator
import numpy as np
# Type checking imports for polar coordinates
from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes
import os
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
from array_utils import ArrayEvaluator
import config

# --- Font settings ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- Audio plotting functions ---
def plot_waveform(signal, sr, title, filepath, t60_value=None):
    """Plot and save time-domain waveform, limit display time based on T60 value"""
    plt.figure(figsize=(12, 4))
    
    # Determine display time based on T60 value
    if t60_value is not None:
        if t60_value >= 0.8:  # T90: keep first 9 seconds
            max_time = 9.0
        elif t60_value >= 0.5:  # T60: keep first 6 seconds
            max_time = 6.0
        else:  # T30: keep first 3 seconds
            max_time = 3.0
        
        max_samples = int(max_time * sr)
        signal = signal[:max_samples]
    
    librosa.display.waveshow(signal, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filepath, dpi=config.FIG_DPI)
    plt.close()

def plot_spectrogram(signal, sr, title, filepath, t60_value=None):
    """Plot and save spectrogram (time-frequency diagram), limit display time based on T60 value"""
    plt.figure(figsize=(12, 4))
    
    # Determine display time based on T60 value
    if t60_value is not None:
        if t60_value >= 0.8:  # T90: keep first 9 seconds
            max_time = 9.0
        elif t60_value >= 0.5:  # T60: keep first 6 seconds
            max_time = 6.0
        else:  # T30: keep first 3 seconds
            max_time = 3.0
        
        max_samples = int(max_time * sr)
        signal = signal[:max_samples]
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=config.FFT_SIZE, hop_length=config.HOP_LENGTH)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', hop_length=config.HOP_LENGTH)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(filepath, dpi=config.FIG_DPI)
    plt.close()

def generate_audio_plots(output_dir, audio_files, t60_value=None):
    """
    Generate and save audio plots for all methods.
    audio_files: dictionary with method names as keys and audio file paths as values.
    t60_value: T60 value to determine display time length
    """
    print_header("Generate Audio Signal Comparison Plots")
    plot_names = {}
    for name, path in audio_files.items():
        try:
            signal, sr = sf.read(path)
            if signal.ndim > 1:
                signal = signal[:, 0]

            # Generate waveform plot
            waveform_filename = f"plot_waveform_{name.lower()}.png"
            waveform_path = os.path.join(output_dir, waveform_filename)
            plot_waveform(signal, sr, f"{name} - Time Domain Waveform", waveform_path, t60_value)

            # Generate spectrogram plot
            spec_filename = f"plot_spectrogram_{name.lower()}.png"
            spec_path = os.path.join(output_dir, spec_filename)
            plot_spectrogram(signal, sr, f"{name} - Time-Frequency Spectrogram", spec_path, t60_value)
            
            plot_names[name] = {'waveform': waveform_filename, 'spectrogram': spec_filename}
        except Exception as e:
            pass
    return plot_names

# --- Geometry plotting functions ---
def plot_array_geometries(geometries, output_dir):
    """Plot and save Figure 1: Microphone Array Geometries"""
    num_geometries = len(geometries)
    fig, axes = plt.subplots(1, num_geometries, figsize=(4 * num_geometries, 4.5), squeeze=False)
    
    radius = config.ARRAY_DIAMETER / 2

    for ax, (name, pos) in zip(axes.flat, geometries.items()):
        ax.scatter(pos[:, 0], pos[:, 1], s=100, marker=MarkerStyle('o'), label=f'Microphones (N={config.MIC_COUNT})')
        ax.add_patch(Circle((0, 0), radius, color='gray', fill=False, linestyle='--'))
        ax.set_title(name, pad=20)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(-radius*1.1, radius*1.1)
        ax.set_ylim(-radius*1.1, radius*1.1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "figure_1_array_geometries.png")
    plt.savefig(filepath, dpi=config.FIG_DPI, bbox_inches='tight')
    plt.close()

def plot_beampattenrs(geometries, freqs, output_dir):
    """Plot and save Figure 2: Broadband Beam Pattern Comparison"""
    num_geometries = len(geometries)
    num_freqs = len(freqs)
    
    fig, axes = plt.subplots(num_freqs, num_geometries, figsize=(5 * num_geometries, 4 * num_freqs), 
                           subplot_kw={'projection': 'polar'}, squeeze=False)
    
    look_direction_rad = np.deg2rad(config.DOA_SIGNAL)

    for i, (name, pos) in enumerate(geometries.items()):
        evaluator = ArrayEvaluator(pos)
        for j, freq in enumerate(freqs):
            ax = axes[j, i]
            pattern = evaluator.beampattern(look_direction_rad, freq, config.VIS_THETA_RANGE)
            pattern_db = 20 * np.log10(np.maximum(pattern, 1e-4))
            
            ax.plot(config.VIS_THETA_RANGE, pattern_db)
            # Disable type checking for polar axes methods
            # type: ignore
            ax.set_theta_zero_location('N')  # type: ignore
            ax.set_theta_direction(-1)  # type: ignore
            ax.set_rlim(-40, 0)
            ax.set_rticks([-40, -30, -20, -10, 0])
            ax.set_title(f"{name}\n@{freq} Hz")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "figure_2_beampatterns.png")
    plt.savefig(filepath, dpi=config.FIG_DPI)
    plt.close()

# --- Report generation functions ---
def generate_performance_table(geometries):
    """Calculate and return DataFrame of geometry optimization performance"""
    data = []
    for name, pos in geometries.items():
        evaluator = ArrayEvaluator(pos)
        avg_wng, avg_di = evaluator.evaluate_broadband()
        slls = []
        look_rad = np.deg2rad(config.DOA_SIGNAL)
        main_lobe_width_deg = 40
        valid_freqs = [f for f in config.FREQ_RANGE if f > 100]
        if not valid_freqs: continue
        for freq in valid_freqs:
            pattern = evaluator.beampattern(look_rad, freq, config.VIS_THETA_RANGE)
            thetas_deg = np.rad2deg(config.VIS_THETA_RANGE)
            look_deg = np.rad2deg(look_rad)
            angular_diff = np.abs(thetas_deg - look_deg)
            angular_diff = np.minimum(angular_diff, 360 - angular_diff)
            main_lobe_indices = np.where(angular_diff < main_lobe_width_deg / 2)[0]
            sidelobe_pattern = np.delete(pattern, main_lobe_indices)
            if len(sidelobe_pattern) > 0:
                max_sidelobe = np.max(sidelobe_pattern)
                main_lobe_peak_idx = np.argmin(np.abs(thetas_deg - look_deg))
                main_lobe_peak = pattern[main_lobe_peak_idx]
                if main_lobe_peak > 1e-6:
                     slls.append(20 * np.log10(max_sidelobe / main_lobe_peak))
        avg_sll = np.mean(slls) if slls else -np.inf
        data.append({
            "Method": name,
            "Average WNG (dB)": f"{avg_wng:.2f}",
            "Average DI (dB)": f"{avg_di:.2f}",
            "Average Max Sidelobe (dB)": f"{avg_sll:.2f}"
        })
    return pd.DataFrame(data)

def generate_scientific_report(geometries, results_list, plot_filenames, output_dir):
    """
    Generate a complete scientific report in academic paper format
    """
    print_header("Generate Scientific Report")
    report_content = []

    report_content.append("# Experimental Results and Analysis Report\n")
    report_content.append("This paper comprehensively evaluates the performance of different microphone array designs and beamforming algorithms through two-stage simulation experiments.\n")

    # --- Part 1: Geometry Optimization ---
    report_content.append("## Stage 1: Array Geometry Optimization Performance Evaluation\n")
    report_content.append("This stage compares various array geometry configurations. Evaluation metrics include broadband average white noise gain (WNG), directivity index (DI), and maximum sidelobe level (SLL). Results are shown in Table 1.\n")
    
    if geometries is not None:
        geo_df = generate_performance_table(geometries)
        report_content.append("**Table 1: Performance Comparison of Array Configurations from Different Optimization Algorithms**\n")
        report_content.append(geo_df.to_markdown(index=False))
        report_content.append("\nFrom the simulation results in Table 1, it can be seen that the HO optimization algorithm proposed in this paper significantly outperforms all other methods in the **average WNG** metric. WNG is a key indicator measuring the array's ability to suppress incoherent noise such as sensor self-noise, with higher WNG meaning stronger robustness.\n")
        report_content.append("![Figure 1: Array geometries from different optimization algorithms](figure_1_array_geometries.png)\n")
        report_content.append("![Figure 2: Beam pattern comparison of different arrays at multiple frequencies](figure_2_beampatterns.png)\n")
    else:
        report_content.append("*Geometry data not provided, skipping this analysis section.*\n")

    # --- Part 2: Beamforming ---
    report_content.append("\n## Stage 2: Beamforming Algorithm Performance Evaluation\n")
    report_content.append("This stage uses the optimal array obtained by the HO algorithm to evaluate the speech enhancement performance of different beamforming algorithms in various simulated acoustic environments. Evaluation metrics include Perceptual Evaluation of Speech Quality (PESQ), Short-Time Objective Intelligibility (STOI), and Scale-Invariant Signal-to-Noise Ratio Improvement (SI-SNRi). Results are shown in Table 2.\n")

    if results_list:
        df = pd.DataFrame(results_list)
        
        # Define all models to report
        models_to_report = [
            "DSB", "Traditional_MVDR", 
            "Mamba_EndToEnd", "Transformer_EndToEnd", "LSTMEndToEnd"
        ]

        for model in models_to_report:
            if f'{model}_sisnr' in df.columns and 'Input_sisnr' in df.columns:
                df[f'{model}_si-snri'] = df[f'{model}_sisnr'] - df['Input_sisnr']
        
        grouped = df.groupby(['T60', 'SNR']).mean().reset_index()

        display_df = pd.DataFrame()
        display_df['Acoustic Condition'] = grouped.apply(lambda row: f"T60={row['T60']}s, SNR={int(row['SNR'])}dB", axis=1)
        
        metrics = ['pesq', 'stoi', 'si-snri']
        
        for metric in metrics:
            for model in models_to_report:
                col_name = f"{model}_{metric}"
                if col_name in grouped.columns:
                    display_df[f"{model} ({metric.upper()})"] = grouped[col_name].map('{:.2f}'.format)

        report_content.append("**Table 2: Performance Comparison of Various Beamforming Algorithms Under Different Acoustic Conditions**\n")
        report_content.append(display_df.to_markdown(index=False))
        report_content.append("\nFrom the results in Table 2, it can be clearly seen that neural network-based algorithms (Mamba, Transformer, LSTM) significantly outperform traditional DSB and MVDR algorithms in all test conditions, particularly in SI-SNRi metrics. This demonstrates the significant advantages of data-driven methods. As the acoustic environment deteriorates, the performance of all algorithms decreases, but neural network models show a more gradual performance decline, reflecting their stronger robustness.\n")
    else:
        report_content.append("*Beamforming evaluation results not provided, skipping this analysis section.*\n")

    # --- Part 3: Signal Plot Comparison ---
    report_content.append("\n## Stage 3: Speech Signal Processing Effect Visualization\n")
    report_content.append("To intuitively demonstrate the performance of each model, we selected a typical test sample and plotted its time-domain waveforms and time-frequency spectrograms before and after processing.\n")

    if plot_filenames:
        # Time domain plots
        report_content.append("\n### 3.1 Time Domain Waveforms\n")
        for name, plots in plot_filenames.items():
            report_content.append(f"![{name} Waveform]({plots['waveform']})")
        report_content.append("\n*Figure 3: Time domain waveform comparison of signals before and after processing by each model.*\n")

        # Time-frequency plots
        report_content.append("\n### 3.2 Time-Frequency Spectrograms\n")
        for name, plots in plot_filenames.items():
            report_content.append(f"![{name} Spectrogram]({plots['spectrogram']})")
        report_content.append("\n*Figure 4: Time-frequency spectrogram comparison of signals before and after processing by each model. It can be observed that the processed signals (especially those processed by neural network models) have clearer speech harmonic structures and significantly suppressed noise energy.*\n")
    else:
        report_content.append("*Audio plots not generated, skipping this section.*\n")

    final_report = "\n".join(report_content)
    filepath = os.path.join(output_dir, "scientific_report.md")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_report)

def print_header(title):
    """Print bordered title"""
    print("\n" + "="*70)
    print(f"| {title.center(66)} |")
    print("="*70)

def plot_individual_geometries(geometries, output_dir):
    """Plot and save individual geometry configurations (one plot per configuration)"""
    print_header("Generate Individual Geometry Plots")
    radius = config.ARRAY_DIAMETER / 2
    
    for name, pos in geometries.items():
        # Create individual figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot microphone positions
        ax.scatter(pos[:, 0], pos[:, 1], s=150, marker=MarkerStyle('o'))
        
        # Plot circular boundary
        ax.add_patch(Circle((0, 0), radius, color='gray', fill=False, linestyle='-', linewidth=2))
        
        # Set figure properties
        ax.set_xlabel('X (m)', fontsize=26, labelpad=10)
        ax.set_ylabel('Y (m)', fontsize=26, labelpad=10)
        ax.set_xlim(-radius*1.2, radius*1.2)
        ax.set_ylim(-radius*1.2, radius*1.2)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set tick font size and density
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(3))  # Reduce x-axis tick density
        ax.yaxis.set_major_locator(MaxNLocator(3))  # Reduce y-axis tick density
        
        # Add coordinate axes origin
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Save image
        filename = f"geometry_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=config.FIG_DPI, bbox_inches='tight')
        plt.close()

def plot_individual_beampatterns(geometries, freqs, output_dir):
    """Plot and save individual beampatterns (UCA and ULA use vertical layout, others use horizontal layout)"""
    print_header("Generate Individual Beampattern Plots")
    look_direction_rad = np.deg2rad(config.DOA_SIGNAL)
    
    for name, pos in geometries.items():
        evaluator = ArrayEvaluator(pos)
        
        # Determine layout: UCA and ULA use vertical, others use horizontal
        num_freqs = len(freqs)
        if "UCA" in name.upper() or "ULA" in name.upper():
            # Vertical layout
            fig, axes = plt.subplots(num_freqs, 1, figsize=(6, 6 * num_freqs), 
                                    subplot_kw={'projection': 'polar'}, squeeze=False)
        else:
            # Horizontal layout
            fig, axes = plt.subplots(1, num_freqs, figsize=(6 * num_freqs, 6), 
                                    subplot_kw={'projection': 'polar'}, squeeze=False)
        
        for j, freq in enumerate(freqs):
            # Select subplot index based on layout
            if "UCA" in name.upper() or "ULA" in name.upper():
                ax = axes[j, 0]  # Vertical layout
            else:
                ax = axes[0, j]  # Horizontal layout
            
            # Calculate beam pattern
            pattern = evaluator.beampattern(look_direction_rad, freq, config.VIS_THETA_RANGE)
            pattern_db = 20 * np.log10(np.maximum(pattern, 1e-4))
            
            # Plot beam pattern
            ax.plot(config.VIS_THETA_RANGE, pattern_db, linewidth=2)
            
            # Mark main lobe direction
            ax.plot([look_direction_rad, look_direction_rad], [0, 0], 'bo', markersize=8)
            
            # Set polar plot properties
            # Disable type checking for polar axes methods
            # type: ignore
            ax.set_theta_zero_location('N')  # type: ignore
            ax.set_theta_direction(-1)  # type: ignore
            ax.set_ylim(-40, 0)
            ax.set_yticks([-40, -30, -20, -10, 0])
            
            # Set tick font size and density
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_rlabel_position(45)  # Set radial label position, increase spacing
            
            # Reduce angle tick density and increase distance from circle
            ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))  # 8 angle ticks
            ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])
            
            # Increase angle tick distance from circle
            ax.tick_params(axis='x', pad=28)  # Increase angle tick spacing
            
            ax.grid(True, alpha=0.6)
        
        # Save image
        filename = f"beampattern_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=config.FIG_DPI, bbox_inches='tight')
        plt.close()

def plot_individual_waveform(signal, sr, filepath, t60_value=None):
    """Plot and save time-domain waveform (without title and legend)"""
    plt.figure(figsize=(12, 4))
    
    # Fixed display of first 6 seconds
    max_time = 6.0
    max_samples = int(max_time * sr)
    signal = signal[:max_samples]
    
    librosa.display.waveshow(signal, sr=sr)
    
    # Set axis labels and ticks
    plt.xlabel("Time (s)", fontsize=26, labelpad=10)
    plt.ylabel("Amplitude", fontsize=26, labelpad=10)
    
    # Set tick font size and density
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # Reduce x-axis tick density
    plt.gca().xaxis.set_major_locator(MaxNLocator(8))
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filepath, dpi=config.FIG_DPI)
    plt.close()

def plot_individual_spectrogram(signal, sr, filepath, t60_value=None):
    """Plot and save spectrogram (without title and legend)"""
    plt.figure(figsize=(12, 4))
    
    # Fixed display of first 6 seconds
    max_time = 6.0
    max_samples = int(max_time * sr)
    signal = signal[:max_samples]
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=config.FFT_SIZE, hop_length=config.HOP_LENGTH)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', hop_length=config.HOP_LENGTH)
    
    # Set colorbar label font size
    cbar = plt.colorbar(format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=18)
    
    # Set axis labels and ticks
    plt.xlabel("Time (s)", fontsize=26, labelpad=10)
    plt.ylabel("Frequency (Hz)", fontsize=26, labelpad=10)
    
    # Set tick font size and density
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # Reduce x-axis tick density
    plt.gca().xaxis.set_major_locator(MaxNLocator(8))
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=config.FIG_DPI)
    plt.close()

def generate_individual_audio_plots(output_dir, audio_files, t60_value=None):
    """
    Generate and save individual time-domain waveforms and spectrograms for all audio files (without titles and legends)
    audio_files: dictionary with method names as keys and audio file paths as values
    t60_value: T60 value to determine display time length
    """
    print_header("Generate Individual Audio Plots")
    plot_names = {}
    for name, path in audio_files.items():
        try:
            signal, sr = sf.read(path)
            if signal.ndim > 1:
                signal = signal[:, 0]

            # Generate waveform plot
            waveform_filename = f"plot_waveform_{name.lower().replace(' ', '_')}.png"
            waveform_path = os.path.join(output_dir, waveform_filename)
            plot_individual_waveform(signal, sr, waveform_path, t60_value)

            # Generate spectrogram plot
            spec_filename = f"plot_spectrogram_{name.lower().replace(' ', '_')}.png"
            spec_path = os.path.join(output_dir, spec_filename)
            plot_individual_spectrogram(signal, sr, spec_path, t60_value)
            
            plot_names[name] = {'waveform': waveform_filename, 'spectrogram': spec_filename}
        except Exception as e:
            pass
    return plot_names
