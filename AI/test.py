#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import gc
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.signal.windows import hann
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Configuration
INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
JOBS = 16
DPI = 300
CUTOFF_FREQ = 3e12  # 3 THz

# Gaussian peak function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def segment_spectra(xf_filtered, spectrum, num_segments):
    """Segment the spectrum into multiple segments."""
    segment_size = len(xf_filtered) // num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else None
        segments.append((xf_filtered[start:end], spectrum[start:end]))
    return segments

def process_file(file_path):
    """Process the .dat file"""
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_prefix = os.path.join(OUTPUT_DIR, base_name)
        print("Processing: %s" % file_path)
        # Read data
        df = pd.read_csv(
            file_path,
            sep=' ',
            usecols=['#', 'Unnamed: 8'],
            dtype={'#': 'int32', 'Unnamed: 8': 'float32'},
            engine='c'
        )
        if df.empty or len(df) < 2:
            raise ValueError("DataFrame is empty or has less than 2 rows.")
        
        # Prepare data
        time = df['#'].values * 2e-3
        signal = df['Unnamed: 8'].values.astype('float32')
        signal -= signal.mean()
        # Plot original data
        plt.figure(figsize=(12, 6))
        plt.plot(time, signal, 'b-', lw=0.8)
        plt.xlabel("Time (ps)")
        plt.ylabel("Dipole moment (D)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("%s_original.png" % output_prefix, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        # Autocorrelation
        n = len(signal)
        fft_sig = rfft(signal, n=2*n)
        autocorr = irfft(fft_sig * np.conj(fft_sig), n=2*n)[:n].real
        autocorr /= np.max(autocorr)
        del fft_sig
        gc.collect()
        # Spectral analysis
        window = hann(n)
        autocorr_windowed = autocorr * window
        yf = rfft(autocorr_windowed)
        xf = rfftfreq(n, d=2e-15)
        # Frequency filtering
        cutoff_idx = np.searchsorted(xf, CUTOFF_FREQ)
        yf[:cutoff_idx] = 0
        # Convert to cm⁻¹
        xf_cm = (xf * 1e-12) / 0.03
        mask = xf_cm <= 4000
        xf_filtered = xf_cm[mask]
        spectrum = 2.0 / n * np.abs(yf[:len(mask)][mask])
        
        # Scale spectrum
        spectrum *= 10000
        
        # Segment the spectrum
        num_segments = 10  # Adjust number of segments as needed
        segments = segment_spectra(xf_filtered, spectrum, num_segments)
        
        # Find peaks in each segment
        peak_frequencies = []
        for xf_segment, spectrum_segment in segments:
            peaks, _ = find_peaks(spectrum_segment, height=np.percentile(spectrum_segment, 95))  # Adjust percentile as needed
            prominences = peak_prominences(spectrum_segment, peaks)[0]
            widths, _, _, _ = peak_widths(spectrum_segment, peaks, rel_height=0.5)
            
            # Filter peaks by prominence and width
            for i, peak in enumerate(peaks):
                if prominences[i] > 50 and widths[i] > 50:  # Adjust thresholds as needed
                    peak_frequencies.append((xf_segment[peak], spectrum_segment[peak]))
        
        # Sort peaks by prominence
        peak_frequencies.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 10 peaks
        selected_peaks = peak_frequencies[:10]
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(xf_filtered, spectrum, 'k-', lw=0.8)
        for freq, amp in selected_peaks:
            plt.scatter(freq, amp, color='red', marker='x', s=100)
        
        # Legend
        legend_labels = [f'{freq:.2f} cm⁻¹' for freq, _ in selected_peaks]
        plt.legend(legend_labels, title="Peaks", loc="upper right")
        
        plt.xlabel("Frequency (cm⁻¹)")
        plt.ylabel("Spectral ACF EDM Amplitude (a. u.)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("%s_spectrum.png" % output_prefix, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print("Error processing %s: %s" % (file_path, str(e)))
        return False

if __name__ == '__main__':
    dat_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith('.dat')
    ]
    if not dat_files:
        print("No .dat files found!")
        exit(1)
    
    print("Number of files found: %d" % len(dat_files))
    print("Processing parameters:")
    print("* Number of cores: %d" % min(JOBS, len(dat_files)))
    print("* Cutoff frequency: %.1f THz" % (CUTOFF_FREQ / 1e12))
    
    with Pool(min(JOBS, len(dat_files))) as pool:
        results = pool.map(process_file, dat_files)
        success_count = sum(results)
        print("Successfully processed: %d/%d" % (success_count, len(dat_files)))