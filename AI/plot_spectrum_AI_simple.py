#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import gc
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from multiprocessing import Pool, cpu_count
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks, peak_widths, peak_prominences


"""
Тут простая обработка: пишем только локальные CSV  в отдельную папку, 
без картинок, без суммарного CSV.

"""

# Configuration
INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.path.abspath(os.path.join(INPUT_DIR, "..", "result"))
JOBS = 16
DPI = 300
CUTOFF_FREQ = 3e12  # 3 THz

def create_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return OUTPUT_DIR

def process_file(file_path, output_dir):
    """Process the .dat file"""
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_prefix = os.path.join(output_dir, base_name)
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
        print(df.head())
        # Prepare data
        time = df['#'].values * 2e-3
        signal = df['Unnamed: 8'].values.astype('float32')
        signal -= signal.mean()
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
        xf = rfftfreq(n, d=1e-15)
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
        # Save spectrum to CSV
        spectrum_df = pd.DataFrame({
            'Frequency_cm-1': xf_filtered,
            'Amplitude': spectrum
        })
        try:
            spectrum_df.to_csv("%s_spectrum.csv" % output_prefix, index=False)
        except Exception as e:
            print(f"Ошибка при сохранении CSV: {e}")
        return True
    except Exception as e:
        print("Error processing %s: %s" % (file_path, str(e)))
        return False

def main():
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
    
    output_dir = create_output_dir()
    
    with Pool(min(JOBS, len(dat_files))) as pool:
        results = pool.starmap(process_file, [(file_path, output_dir) for file_path in dat_files])
        success_count = sum(results)
        print("Successfully processed: %d/%d" % (success_count, len(dat_files)))

if __name__ == '__main__':
    main()