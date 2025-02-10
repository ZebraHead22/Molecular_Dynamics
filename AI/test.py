#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import gc
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.signal.windows import hann
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# Configuration
INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
JOBS = 16
DPI = 300
CUTOFF_FREQ = 3e12  # 3 THz

def detect_peaks(xf_filtered, spectrum, num_segments=8, num_amplitude_segments=5):
    """Detect peaks in the spectrum by dividing into frequency and amplitude segments."""
    if len(xf_filtered) == 0 or len(spectrum) == 0:
        return []
    # Define frequency segments
    frequency_segments = np.linspace(0, 4000, num_segments + 1)
    peaks = []
    for i in range(num_segments):
        seg_start = frequency_segments[i]
        seg_end = frequency_segments[i + 1]
        # Find indices within the current frequency segment
        in_segment = np.where((xf_filtered >= seg_start) & (xf_filtered <= seg_end))[0]
        if len(in_segment) == 0:
            continue
        # Extract the spectrum segment
        spectrum_segment = spectrum[in_segment]
        # Define amplitude segments within the current frequency segment
        amplitude_segments = np.linspace(np.min(spectrum_segment), np.max(spectrum_segment), num_amplitude_segments + 1)
        for j in range(num_amplitude_segments):
            seg_amp_start = amplitude_segments[j]
            seg_amp_end = amplitude_segments[j + 1]
            # Find indices within the current amplitude segment
            in_amp_segment = np.where((spectrum_segment >= seg_amp_start) & (spectrum_segment <= seg_amp_end))[0]
            if len(in_amp_segment) == 0:
                continue
            # Find the maximum in the current amplitude segment
            max_idx = in_segment[in_amp_segment[np.argmax(spectrum_segment[in_amp_segment])]]
            freq = xf_filtered[max_idx]
            amp = spectrum[max_idx]
            peaks.append((freq, amp))
    # Sort peaks by amplitude in descending order
    peaks.sort(key=lambda x: -x[1])
    # Remove peaks that are too close to each other
    filtered_peaks = []
    for freq, amp in peaks:
        if any(abs(freq - f) < 10 for f, _ in filtered_peaks):
            continue
        if amp < max(peaks, key=lambda x: x[1])[1] * 0.03:
            continue
        filtered_peaks.append((freq, amp))
    return filtered_peaks

def process_file(file_path):
    """Process a .dat file to generate spectrum and peak plots."""
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_prefix = os.path.join(OUTPUT_DIR, base_name)
        print(f"Processing: {file_path}")
        # Read data
        df = pd.read_csv(
            file_path,
            sep=' ',
            usecols=['#', 'Unnamed: 8'],
            dtype={'#': 'int32', 'Unnamed: 8': 'float32'},
            engine='c'
        )
        if df.empty or len(df) < 2:
            raise ValueError("DataFrame is empty or has fewer than 2 rows.")
        # Prepare data
        time = df['#'].values * 2e-3  # Convert fs to ps
        signal = df['Unnamed: 8'].values.astype('float32')
        signal -= signal.mean()  # Remove DC offset
        # Plot original signal
        plt.figure(figsize=(12, 6))
        plt.plot(time, signal, 'b-', lw=0.8)
        plt.xlabel("Time (ps)")
        plt.ylabel("Dipole moment (D)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_original.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        # Autocorrelation with FFT
        n = len(signal)
        fft_sig = rfft(signal, n=2*n)
        autocorr = irfft(fft_sig * np.conj(fft_sig), n=2*n)[:n].real
        autocorr /= np.max(autocorr)
        del fft_sig
        gc.collect()
        # Spectral analysis with Hanning window
        window = hann(n)
        autocorr_windowed = autocorr * window
        yf = rfft(autocorr_windowed)
        xf = rfftfreq(n, d=2e-15)  # d in seconds (2 fs)
        # Apply frequency cutoff
        cutoff_idx = np.searchsorted(xf, CUTOFF_FREQ)
        yf[:cutoff_idx] = 0
        # Convert to cm⁻¹ and limit to 4000 cm⁻¹
        xf_cm = (xf * 1e-12) / 0.03  # Convert THz to cm⁻¹
        mask = xf_cm <= 4000
        xf_filtered = xf_cm[mask]
        yf_masked = yf[:len(xf_filtered)]
        spectrum = 2.0 / n * np.abs(yf_masked)
        spectrum *= 10000  # Scaling factor
        # Smooth the spectrum
        smoothed_spectrum = savgol_filter(spectrum, window_length=11, polyorder=2)
        # Detect peaks
        selected_peaks = detect_peaks(xf_filtered, smoothed_spectrum,
                                      num_segments=10,
                                      num_amplitude_segments=5)
         # Plot spectrum with peaks
        plt.figure(figsize=(12, 6))
        plt.plot(xf_filtered, smoothed_spectrum, 'k-', lw=0.8, label='_nolegend_')  # Исключаем линию из легенды
        # Добавляем все пики с метками
        for freq, amp in selected_peaks:
            plt.scatter(freq, amp, color='red', marker='x', s=100, label=f'{freq:.2f} cm⁻¹')
        # Создаем легенду только для пиков
        plt.legend(title=None, loc="upper right")
        plt.xlabel("Frequency (cm⁻¹)")
        plt.ylabel("Spectral ACF EDM Amplitude (a. u.)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_spectrum.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

if __name__ == '__main__':
    dat_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.dat')]
    if not dat_files:
        print("No .dat files found!")
        exit(1)
    print(f"Number of files found: {len(dat_files)}")
    print("Processing parameters:")
    print(f"* Number of cores: {min(JOBS, len(dat_files))}")
    print(f"* Cutoff frequency: {CUTOFF_FREQ / 1e12:.1f} THz")
    with Pool(min(JOBS, len(dat_files))) as pool:
        results = pool.map(process_file, dat_files)
        success_count = sum(results)
        print(f"Successfully processed: {success_count}/{len(dat_files)}")