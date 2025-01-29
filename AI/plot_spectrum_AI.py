import os
import numpy as np
import pandas as pd
import gc
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.signal.windows import hann
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
JOBS = 16
DPI = 300
CUTOFF_FREQ = 3e12  # 3 THz

def find_significant_peaks(spectrum, frequencies):
    # Функция для нахождения пиков в заданном диапазоне частот
    def find_peaks_in_range(freq_range, spectrum, frequencies):
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freq_subset = frequencies[mask]
        spec_subset = spectrum[mask]
        
        # Находим пик
        peak_indices, _ = find_peaks(spec_subset)
        if len(peak_indices) == 0:
            return None, None
        
        # Выбираем пик с наибольшей амплитудой
        max_peak_index = peak_indices[np.argmax(spec_subset[peak_indices])]
        max_freq = freq_subset[max_peak_index]
        max_amp = spec_subset[max_peak_index]
        
        return max_freq, max_amp
    
    # Определяем диапазоны частот
    freq_ranges = [
        (0, 500), (500, 1000), (1000, 1500), (1500, 2000),
        (2000, 2500), (2500, 3000), (3000, 3500), (3500, 4000)
    ]
    
    significant_peaks = []
    max_amp = np.max(spectrum)
    
    for freq_range in freq_ranges:
        max_freq, max_amp_range = find_peaks_in_range(freq_range, spectrum, frequencies)
        if max_amp_range is not None and max_amp_range >= 0.05 * max_amp:
            significant_peaks.append((max_freq, max_amp_range))
    
    # Преобразуем список кортежей в массив индексов пиков
    peak_indices = np.array([np.argmin(np.abs(frequencies - peak[0])) for peak in significant_peaks])
    
    return peak_indices


def process_file(file_path):
    """Process the .dat file"""
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
        xf = rfftfreq(n, d=2e-15)
        # Frequency filtering
        cutoff_idx = np.searchsorted(xf, CUTOFF_FREQ)
        yf[:cutoff_idx] = 0
        # Convert to cm⁻¹
        xf_cm = (xf * 1e-12) / 0.03
        mask = xf_cm <= 4000
        xf_filtered = xf_cm[mask]
        spectrum = 2.0 / n * np.abs(yf[:len(mask)][mask])
        print(f"Spectrum length: {len(spectrum)}, xf_filtered length: {len(xf_filtered)}")
        # Peak search
        peaks = find_significant_peaks(spectrum, xf_filtered)
        
        # Convert peaks to integer indices
        peaks = peaks.astype(int)
        
        # Scale spectrum
        spectrum *= 10000
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(xf_filtered, spectrum, 'k-', lw=0.8)
        if len(peaks) > 0:
            # Exact positioning of markers
            x_peaks = xf_filtered[peaks]
            y_peaks = spectrum[peaks]  # Already scaled values
            # Draw crosses exactly on the peaks
            plt.scatter(
                x_peaks,
                y_peaks,
                marker=r'$\times$',
                color='red',
                s=60,  # Marker size
                linewidth=1.2,
                zorder=3  # Above other elements
            )
            # Legend
            legend_labels = [f"{xf_filtered[peak]:.1f} cm⁻¹" for peak in peaks]
            plt.legend(
                labels=legend_labels,
                loc='upper right',
                frameon=False,
                fontsize=9,
                title='Peak Frequencies:',
                handlelength=0,  # No line handles
                handletextpad=0,  # No padding between text and handle
                borderpad=0,  # No padding around the legend box
                markerscale=0 # Hide markers in legend
            )
        plt.xlabel("Frequency (cm⁻¹)")
        plt.ylabel("Spectral ACF EDM Amplitude (a. u.)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_spectrum.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        
        # Print peak frequencies and amplitudes
        for peak in peaks:
            freq = xf_filtered[peak]
            amp = spectrum[peak]
            print(f"Peak found: Frequency = {freq:.1f} cm⁻¹, Amplitude = {amp:.1f}")
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
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
    
    print(f"Number of files found: {len(dat_files)}")
    print("Processing parameters:")
    print(f"* Number of cores: {min(JOBS, len(dat_files))}")
    print(f"* Cutoff frequency: {CUTOFF_FREQ / 1e12:.1f} THz")
    
    with Pool(min(JOBS, len(dat_files))) as pool:
        results = pool.map(process_file, dat_files)
        success_count = sum(results)
        print(f"Successfully processed: {success_count}/{len(dat_files)}")