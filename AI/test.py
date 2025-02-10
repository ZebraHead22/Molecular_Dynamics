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

# Configuration
INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
JOBS = 16
DPI = 300

def detect_peaks(xf_filtered, spectrum, num_segments=10, top_n=10,)
    pass

def process_file(file_path):
    """Обрабатывает файл .dat."""
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_prefix = os.path.join(OUTPUT_DIR, base_name)
        print("Processing: %s" % file_path)
        
        # Чтение данных
        df = pd.read_csv(
            file_path,
            sep=' ',
            usecols=['#', 'Unnamed: 8'],
            dtype={'#': 'int32', 'Unnamed: 8': 'float32'},
            engine='c'
        )
        if df.empty or len(df) < 2:
            raise ValueError("DataFrame is пустой или содержит менее 2 строк.")
        
        # Подготовка данных
        time = df['#'].values * 2e-3
        signal = df['Unnamed: 8'].values.astype('float32')
        signal -= signal.mean()
        
        # Построение графика исходного сигнала
        plt.figure(figsize=(12, 6))
        plt.plot(time, signal, 'b-', lw=0.8)
        plt.xlabel("Time (ps)")
        plt.ylabel("Dipole moment (D)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_original.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        
        # Автокорреляция
        n = len(signal)
        fft_sig = rfft(signal, n=2*n)
        autocorr = irfft(fft_sig * np.conj(fft_sig), n=2*n)[:n].real
        autocorr /= np.max(autocorr)
        del fft_sig
        gc.collect()
        
        # Спектральный анализ с использованием окна Ханна
        window = hann(n)
        autocorr_windowed = autocorr * window
        yf = rfft(autocorr_windowed)
        xf = rfftfreq(n, d=2e-15)
        
        # Фильтрация по частоте
        cutoff_idx = np.searchsorted(xf, CUTOFF_FREQ)
        yf[:cutoff_idx] = 0
        
        # Перевод частот в см⁻¹
        xf_cm = (xf * 1e-12) / 0.03
        mask = xf_cm <= 4000
        xf_filtered = xf_cm[mask]
        yf_masked = yf[:len(xf_filtered)]
        spectrum = 2.0 / n * np.abs(yf_masked)
        
        # Масштабирование спектра (при необходимости)
        spectrum *= 10000
        
        # Выделение пиков (напр., топ-10)
        # Если слишком много пиков, попробуйте увеличить параметры min_prominence и/или min_width,
        # либо повысить percentile для определения порога высоты.
        selected_peaks = detect_peaks(xf_filtered, spectrum,
                                      num_segments=10,
                                      top_n=10,
                                      height_percentile=95,   # можно снизить до 90
                                      min_prominence=50,       # можно увеличить для уменьшения числа пиков
                                      min_width=50)            # можно изменить в зависимости от ширины пиков
        
        # Построение графика спектра и нанесение красных крестиков для найденных пиков
        plt.figure(figsize=(12, 6))
        plt.plot(xf_filtered, spectrum, 'k-', lw=0.8, label="Spectrum")
        for freq, amp in selected_peaks:
            plt.scatter(freq, amp, color='red', marker='x', s=100)
        # Легенда с частотами пиков
        legend_labels = [f'{freq:.2f} cm⁻¹' for freq, _ in selected_peaks]
        plt.legend(legend_labels, title="Peaks", loc="upper right")
        plt.xlabel("Frequency (cm⁻¹)")
        plt.ylabel("Spectral ACF EDM Amplitude (a. u.)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_spectrum.png", dpi=DPI, bbox_inches='tight')
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
