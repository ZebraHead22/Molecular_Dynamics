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

"""
Тут делаем все: и суммарные файлы, и частные файлы, и графики для каждого файла.
"""

# Configuration
INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
JOBS = 16
DPI = 300
CUTOFF_FREQ = 3e12  # 3 THz

def detect_peaks(xf_filtered, smoothed_spectrum, original_spectrum):
    """Detect peaks using sliding window and iterative filtering."""
    max_amp = np.max(original_spectrum)
    # Step 1: Find all local maxima with 3-point window
    peaks = []
    for i in range(1, len(smoothed_spectrum) - 1):
        if smoothed_spectrum[i] > smoothed_spectrum[i - 1] and smoothed_spectrum[i] > smoothed_spectrum[i + 1]:
            peaks.append((xf_filtered[i], smoothed_spectrum[i]))
    if not peaks:
        return []
    # Step 2: Iteratively filter until less than 20 peaks remain
    current_peaks = peaks.copy()
    while len(current_peaks) >= 1000:
        new_peaks = []
        n = len(current_peaks)
        if n == 0:
            break
        for i in range(n):
            if n == 1:
                new_peaks.append(current_peaks[i])
                break
            if i == 0:
                if current_peaks[i][1] > current_peaks[i + 1][1]:
                    new_peaks.append(current_peaks[i])
            elif i == n - 1:
                if current_peaks[i][1] > current_peaks[i - 1][1]:
                    new_peaks.append(current_peaks[i])
            else:
                if current_peaks[i][1] > current_peaks[i - 1][1] and current_peaks[i][1] > current_peaks[i + 1][1]:
                    new_peaks.append(current_peaks[i])
        if len(new_peaks) == len(current_peaks):
            break
        current_peaks = new_peaks
    # Filter peaks below 3% of max amplitude from original spectrum
    current_peaks = [peak for peak in current_peaks if peak[1] >= 0.015 * max_amp]
    return current_peaks

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
        # Smooth spectrum
        smoothed_spectrum = savgol_filter(spectrum, window_length=11, polyorder=2)
        # Detect peaks with original spectrum for amplitude threshold
        selected_peaks = detect_peaks(xf_filtered, smoothed_spectrum, spectrum)
        # Extract and sort frequencies
        freq_list = sorted([peak[0] for peak in selected_peaks])
        
        # Save spectrum to CSV
        spectrum_df = pd.DataFrame({
            'Frequency_cm-1': xf_filtered,
            'Amplitude': spectrum
        })
        spectrum_df.to_csv("%s_spectrum.csv" % output_prefix, index=False)
        
        # Plot spectrum with peaks
        plt.figure(figsize=(12, 6))
        plt.plot(xf_filtered, smoothed_spectrum, 'k-', lw=0.8, label='_nolegend_')
        for freq, amp in selected_peaks:
            plt.scatter(freq, amp, color='red', marker='x', s=100, label=f'{freq:.2f} cm⁻¹')
        plt.legend(title=None, loc="upper right")
        plt.xlabel("Frequency (cm⁻¹)")
        plt.ylabel("Spectral ACF EDM Amplitude (a. u.)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_spectrum.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        return freq_list
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

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
        # Collect successful results
        success_files = []
        all_peaks = []
        for file_path, peaks in zip(dat_files, results):
            if isinstance(peaks, list) and len(peaks) > 0:
                success_files.append(os.path.basename(file_path))
                all_peaks.append(peaks)
        success_count = len(success_files)
        print(f"Successfully processed: {success_count}/{len(dat_files)}")
        # Generate CSV report
        if success_count > 0:
            max_peaks = max(len(peaks) for peaks in all_peaks)
            # Pad each peak list to max_peaks length
            padded_peaks = [peaks + [''] * (max_peaks - len(peaks)) for peaks in all_peaks]
            # Transpose rows and columns
            transposed = list(zip(*padded_peaks))
            # Format values to strings with 2 decimal places
            formatted_transposed = []
            for row in transposed:
                formatted_row = []
                for val in row:
                    if isinstance(val, float):
                        formatted_row.append(f"{val:.2f}")
                    else:
                        formatted_row.append(str(val))
                formatted_transposed.append(formatted_row)
            # Write CSV
            csv_path = os.path.join(OUTPUT_DIR, "peaks_summary.csv")
            with open(csv_path, 'w', encoding='utf-8') as f:
                # Header with filenames
                f.write(';'.join(success_files) + '\n')
                # Write each transposed row
                for row in formatted_transposed:
                    f.write(';'.join(row) + '\n')
            print(f"CSV report saved to: {csv_path}")
        else:
            print("No peaks detected in any files, CSV report skipped.")