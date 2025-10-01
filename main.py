import numpy as np
import os
import argparse
import soundfile as sf
from tqdm import tqdm
import glob
import torch
import sys

import config
from optimizers import run_all_optimizers
from acoustic_simulation import generate_dataset
from visualization import (
    generate_scientific_report, 
    plot_array_geometries, 
    plot_beampattenrs, 
    generate_performance_table,
    generate_audio_plots
)
from beamformers import (
    delay_and_sum, 
    traditional_mvdr, 
    end_to_end_enhancer
)
from process_data import preprocess_data_pipeline

def run_geometry_optimization():
    """Execute Phase 1: Array geometry optimization"""
    print_header("Phase 1: Array Geometry Optimization")
    radius = config.ARRAY_DIAMETER / 2
    ula_pos = np.zeros((config.MIC_COUNT, 2))
    ula_pos[:, 0] = np.linspace(-radius, radius, config.MIC_COUNT)
    uca_pos = np.zeros((config.MIC_COUNT, 2))
    angles = np.linspace(0, 2 * np.pi, config.MIC_COUNT, endpoint=False)
    uca_pos[:, 0], uca_pos[:, 1] = radius * np.cos(angles), radius * np.sin(angles)

    geometries = {
        "Uniform Linear Array (ULA)": ula_pos, 
        "Uniform Circular Array (UCA)": uca_pos
    }
    optimized = run_all_optimizers()
    geometries.update(optimized)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plot_array_geometries(geometries, config.OUTPUT_DIR)
    plot_beampattenrs(geometries, freqs=[500, 2000, 6000], output_dir=config.OUTPUT_DIR)
    
    geo_df = generate_performance_table(geometries)
    print("\n--- Table 1: Performance Comparison of Optimized Array Configurations ---\n")
    print(geo_df.to_markdown(index=False))
    
    return geometries, geometries["HO Optimized Array"]

def run_evaluation(best_geometry, geometries_for_report):
    """Evaluate all methods on test set and generate scientific comparison report"""
    print_header("Final Evaluation: Scientific Comparison of All Methods")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError:
        print("Error: Please install evaluation libraries: pip install pesq pystoi")
        return

    test_files = [f.replace("_mixed.wav", "") for f in sorted(glob.glob(os.path.join(config.TEST_DIR, "*_mixed.wav")))]
    results_list = []

    methods_to_eval = {
        # 1. Traditional baseline methods
        "DSB": { "method": lambda sig, geo, doa: delay_and_sum(sig, geo, doa), "requires_doa": True },
        "Traditional_MVDR": { "method": lambda sig, geo, doa: traditional_mvdr(sig, geo, doa), "requires_doa": True },
        
        # 2. End-to-end neural network methods
        "Mamba_EndToEnd": { "method": lambda sig, geo, doa: end_to_end_enhancer(sig, "mamba_end_to_end", device), "requires_doa": False },
        "Transformer_EndToEnd": { "method": lambda sig, geo, doa: end_to_end_enhancer(sig, "transformer_end_to_end", device), "requires_doa": False },
        "LSTMEndToEnd": { "method": lambda sig, geo, doa: end_to_end_enhancer(sig, "lstm_end_to_end", device), "requires_doa": False }
    }
    
    # --- Audio file management for plotting ---
    audio_output_dir = os.path.join(config.OUTPUT_DIR, 'audio_plots')
    os.makedirs(audio_output_dir, exist_ok=True)
    plot_sample_basename = os.path.basename(test_files[0]) if test_files else None
    audio_files_for_plotting = {}
    
    for path in tqdm(test_files, desc="Evaluating all methods"):
        mixed_sig, sr = sf.read(f"{path}_mixed.wav", dtype='float32')
        clean_sig, _ = sf.read(f"{path}_clean.wav", dtype='float32')
        if clean_sig.ndim > 1: clean_sig = clean_sig[:, 0]
            
        is_plot_sample = (os.path.basename(path) == plot_sample_basename)
        if is_plot_sample:
            input_path = os.path.join(audio_output_dir, "Input.wav")
            clean_path = os.path.join(audio_output_dir, "Clean_Target.wav")
            sf.write(input_path, mixed_sig[:, 0], sr)
            sf.write(clean_path, clean_sig, sr)
            audio_files_for_plotting["Input (Mixed Signal)"] = input_path
            audio_files_for_plotting["Clean (Target)"] = clean_path

        t60_value = float(np.random.choice(config.SIM_T60_CONDITIONS))
        current_results = {
            "file": os.path.basename(path),
            "T60": t60_value,
            "SNR": float(np.random.choice(config.SIM_SNR_CONDITIONS))
        }

        def calculate_metrics(clean, enhanced, sr):
            min_len = min(len(clean), len(enhanced))
            clean, enhanced = clean[:min_len], enhanced[:min_len]
            try:
                pesq_score = pesq(sr, clean, enhanced, 'wb') if sr == 16000 else 1.0
            except Exception: pesq_score = 1.0
            stoi_score = stoi(clean, enhanced, sr, extended=False)
            s_target = (np.dot(enhanced, clean) / (np.dot(clean, clean) + 1e-8)) * clean
            e_noise = enhanced - s_target
            si_snr = 10 * np.log10((np.dot(s_target, s_target) + 1e-8) / (np.dot(e_noise, e_noise) + 1e-8))
            return pesq_score, stoi_score, si_snr

        ref_mic_sig = mixed_sig[:, 0]
        pesq_in, stoi_in, sisnr_in = calculate_metrics(clean_sig, ref_mic_sig, sr)
        current_results.update({"Input_pesq": pesq_in, "Input_stoi": stoi_in, "Input_sisnr": sisnr_in})
        
        for name, method_info in methods_to_eval.items():
            try:
                enhanced_sig = method_info["method"](mixed_sig.T, best_geometry, config.DOA_SIGNAL)
                
                if is_plot_sample:
                    enhanced_path = os.path.join(audio_output_dir, f"{name}.wav")
                    sf.write(enhanced_path, enhanced_sig, sr)
                    audio_files_for_plotting[name] = enhanced_path

                pesq_out, stoi_out, sisnr_out = calculate_metrics(clean_sig, enhanced_sig, sr)
                current_results.update({f"{name}_pesq": pesq_out, f"{name}_stoi": stoi_out, f"{name}_sisnr": sisnr_out})
                
            except FileNotFoundError as e:
                current_results.update({f"{name}_pesq": 1.0, f"{name}_stoi": 0.0, f"{name}_sisnr": -100.0})
            except Exception as e:
                current_results.update({f"{name}_pesq": 1.0, f"{name}_stoi": 0.0, f"{name}_sisnr": -100.0})
        
        results_list.append(current_results)
    
    # --- Generate all charts and reports ---
    plot_filenames = generate_audio_plots(config.OUTPUT_DIR, audio_files_for_plotting, t60_value)
    print("\nGenerating scientific comparison report...")
    generate_scientific_report(geometries_for_report, results_list, plot_filenames, config.OUTPUT_DIR)

def print_header(title):
    print("\n" + "="*70)
    print(f"| {title.center(66)} |")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Modular experimental workflow coordinator")
    subparsers = parser.add_subparsers(dest='stage', required=True, help="Select experiment stage to run")

    subparsers.add_parser('opt', help="Array geometry optimization")
    
    datagen_parser = subparsers.add_parser('datagen', help="Generate acoustic training data")
    datagen_parser.add_argument('--data-type', choices=['train', 'test'], default=None, help="Specify to generate train or test data (default: generate all)")
    datagen_parser.add_argument('--num-samples', type=int, default=None, help="Specify number of samples to generate (default: use config.py settings)")
    datagen_parser.add_argument('--start-index', type=int, default=0, help="Starting index for filenames (default: 0)")
    
    subparsers.add_parser('preprocess', help="Preprocess training data")
    
    train_parser = subparsers.add_parser('train', help="Train neural network models")
    train_parser.add_argument('--model', choices=['all', 'mamba', 'transformer', 'lstm'], default='all', help="Specify model to train (default: all)")
    
    subparsers.add_parser('evaluate', help="Full method performance evaluation and comparison")
    
    subparsers.add_parser('drawgeo', help="Plot saved geometries (individual images)")
    
    subparsers.add_parser('drawwav', help="Plot audio file waveforms and spectrograms (individual images)")

    args = parser.parse_args()

    geo_file_path = os.path.join(config.OUTPUT_DIR, "best_ho_geometry.npy")
    all_geo_file_path = os.path.join(config.OUTPUT_DIR, "all_geometries.npy")

    if args.stage == 'opt':
        geometries_for_report, best_ho_geometry = run_geometry_optimization()
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(geo_file_path, 'wb') as f: np.save(f, best_ho_geometry)
        with open(all_geo_file_path, 'wb') as f: np.save(f, np.array(list(geometries_for_report.items()), dtype=object))
        print("\n[Opt] Stage completed.")
        
    elif args.stage == 'datagen':
        try:
            with open(geo_file_path, 'rb') as f: best_ho_geometry = np.load(f)
            print_header("Generating Acoustic Dataset (WAV)")
            if args.data_type is None:
                train_samples = args.num_samples if args.num_samples else config.NUM_TRAIN_SAMPLES
                test_samples = args.num_samples if args.num_samples else config.NUM_TEST_SAMPLES
                generate_dataset(best_ho_geometry, 'train', train_samples, args.start_index)
                generate_dataset(best_ho_geometry, 'test', test_samples, args.start_index)
            else:
                num_samples = args.num_samples if args.num_samples else (config.NUM_TRAIN_SAMPLES if args.data_type == 'train' else config.NUM_TEST_SAMPLES)
                generate_dataset(best_ho_geometry, args.data_type, num_samples, args.start_index)
            print("\n[Datagen] Stage completed.")
        except FileNotFoundError:
            print(f"Error: Geometry file '{geo_file_path}' not found. Please run 'python main.py opt' first.")
            
    elif args.stage == 'preprocess':
        print_header("Data Preprocessing")
        preprocess_data_pipeline('end_to_end')
        print("\n[Preprocess] Stage completed.")
            
    elif args.stage == 'train':
        print_header("Training Neural Network Models")
        try:
            from train import train
            
            # Determine model configurations based on --model parameter
            if args.model == 'all':
                model_configs = [
                    ('end_to_end', 'mamba'),
                    ('end_to_end', 'transformer'),
                    ('end_to_end', 'lstm')
                ]
                print("Training all models")
            else:
                model_configs = [('end_to_end', args.model)]
                print(f"Training specified model: {args.model}")
            
            for model_type, arch in model_configs:
                print(f"\nStarting training for model: {arch}_{model_type}")
                try:
                    train(model_type=model_type, arch=arch)
                except Exception as e:
                    print(f"Error training model {arch}_{model_type}: {str(e)}")
                    continue
            print("\n[Train] Stage completed.")
        except ImportError:
            print("Error: train.py not found. Please ensure the file exists.")

    elif args.stage == 'evaluate':
        geometries_for_report, best_ho_geometry = None, None
        try:
            with open(geo_file_path, 'rb') as f: best_ho_geometry = np.load(f)
        except FileNotFoundError:
            print(f"Error: Core geometry file '{geo_file_path}' not found. Please run 'opt' stage first.")
            sys.exit(1)
        try:
            with open(all_geo_file_path, 'rb') as f:
                geometries_for_report = dict(np.load(f, allow_pickle=True))
        except FileNotFoundError:
            print(f"Warning: Comparison geometry file '{all_geo_file_path}' not found. Report will lack geometry comparison section.")
        run_evaluation(best_ho_geometry, geometries_for_report)
        print("\n[Evaluate] Stage completed.")
        
    elif args.stage == 'drawgeo':
        print_header("Plotting Saved Geometries (Individual Images)")
        try:
            with open(all_geo_file_path, 'rb') as f:
                geometries = dict(np.load(f, allow_pickle=True))
            print(f"Successfully loaded {len(geometries)} geometries")
            
            # Import new plotting functions
            from visualization import plot_individual_geometries, plot_individual_beampatterns
            
            # Create output directory for individual images
            individual_output_dir = os.path.join(config.OUTPUT_DIR, "individual_plots")
            os.makedirs(individual_output_dir, exist_ok=True)
            
            # Plot individual geometries
            plot_individual_geometries(geometries, individual_output_dir)
            
            # Plot individual beam patterns
            plot_individual_beampatterns(geometries, freqs=[500, 2000, 6000], output_dir=individual_output_dir)
            
            print(f"\n[Drawgeo] Stage completed. All images saved to {individual_output_dir}")
            
        except FileNotFoundError:
            print(f"Error: Geometry file '{all_geo_file_path}' not found. Please run 'opt' stage first.")
        except ImportError:
            print("Error: New plotting functions not found. Please ensure visualization.py is updated.")
            
    elif args.stage == 'drawwav':
        print_header("Plotting Audio File Waveforms and Spectrograms (Individual Images)")
        audio_plots_dir = os.path.join(config.OUTPUT_DIR, "audio_plots")
        
        if not os.path.exists(audio_plots_dir):
            print(f"Error: Audio file directory '{audio_plots_dir}' not found. Please run 'evaluate' stage first to generate audio files.")
            sys.exit(1)
        
        # Get all audio files
        audio_files = {}
        for wav_file in glob.glob(os.path.join(audio_plots_dir, "*.wav")):
            name = os.path.splitext(os.path.basename(wav_file))[0]
            # Convert filenames to more friendly display names
            if name == "Input":
                display_name = "Input (Mixed Signal)"
            elif name == "Clean_Target":
                display_name = "Clean (Target)"
            else:
                display_name = name.replace("_", " ")
            audio_files[display_name] = wav_file
        
        if not audio_files:
            print(f"Error: No audio files found in directory '{audio_plots_dir}'.")
            sys.exit(1)
        
        print(f"Found {len(audio_files)} audio files:")
        for name in audio_files.keys():
            print(f"  - {name}")
        
        # Import new plotting function
        from visualization import generate_individual_audio_plots
        
        # Create output directory for individual audio images
        individual_audio_output_dir = os.path.join(config.OUTPUT_DIR, "individual_audio_plots")
        os.makedirs(individual_audio_output_dir, exist_ok=True)
        
        # Generate individual audio plots
        plot_filenames = generate_individual_audio_plots(individual_audio_output_dir, audio_files)
        
        print(f"\n[Drawwav] Stage completed. All audio images saved to {individual_audio_output_dir}")
        print("Generated images include:")
        for name, plots in plot_filenames.items():
            print(f"  - {name}: {plots['waveform']}, {plots['spectrogram']}")

if __name__ == "__main__":
    main()
