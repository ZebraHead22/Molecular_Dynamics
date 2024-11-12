import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hann, correlate
from multiprocessing import Pool, cpu_count

def create_title(filename):
    match = re.search(r'(\w+)_(\d+)_([a-zA-Z]+)', filename)
    if match:
        prefix = match.group(1).upper()
        number = match.group(2)
        environment = match.group(3).lower()

        if 'water' in environment:
            return f"{prefix} WATER N={number}"
        elif 'vac' in environment or 'vacuum' in environment:
            return f"{prefix} VACUUM N={number}"
        elif 'linear' in environment:
            return f"{prefix} LINEAR N={number}"
        elif 'cyclic' in environment:
            return f"{prefix} CYCLIC N={number}"
    return filename

def compute_spectrum(dipole_moment, time_step):
    # Calculate autocorrelation
    autocorr = correlate(dipole_moment, dipole_moment, mode='full')
    autocorr = autocorr[autocorr.size // 2:]  # take only positive lags

    # Apply Hann window
    hann_window = hann(len(autocorr))
    autocorr_windowed = autocorr * hann_window

    # Perform FFT on the windowed autocorrelation
    freq = fftfreq(len(autocorr_windowed), d=time_step)
    spectrum = np.abs(fft(autocorr_windowed))  # magnitude of the spectrum

    # Return only positive frequencies and corresponding spectrum values
    positive_freqs = freq[freq >= 0]
    positive_spectrum = spectrum[freq >= 0]

    return positive_freqs, positive_spectrum

def plot_spectra(positive_freqs, spectra, title, colors, labels):
    plt.figure(figsize=(12, 8))
    freq_ranges = [(0, 1000), (1000, 2000), (2000, 3000), 
                   (3000, 4000), (4000, 5000), (5000, 6000)]
    
    for i, (f_min, f_max) in enumerate(freq_ranges):
        plt.subplot(2, 3, i + 1)
        
        # Plot each spectrum within the given frequency range
        for spectrum, color, label in zip(spectra, colors, labels):
            mask = (positive_freqs >= f_min) & (positive_freqs < f_max)
            plt.plot(positive_freqs[mask], spectrum[mask], color=color, label=label)

        plt.xlabel("Frequency (cm⁻¹)")
        plt.ylabel("Amplitude")
        plt.title(f"Frequency range {f_min}-{f_max} cm⁻¹")
        plt.legend()
        plt.grid()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def process_pair(file1, file2, amino_acid):
    try:
        # Load data from the .dat files
        df1 = pd.read_csv(file1, sep=' ')
        df2 = pd.read_csv(file2, sep=' ')

        # Rename and prepare data
        df1.dropna(how='all', axis=1, inplace=True)
        df1.rename(columns={'#': 'frame', 'Unnamed: 8': '|dip|'}, inplace=True)

        df2.dropna(how='all', axis=1, inplace=True)
        df2.rename(columns={'#': 'frame', 'Unnamed: 8': '|dip|'}, inplace=True)

        time_step = 2e-6  # time in ns, assuming 2 fs per frame

        # Compute spectra for both files
        freqs1, spectrum1 = compute_spectrum(np.array(df1["|dip|"]), time_step)
        freqs2, spectrum2 = compute_spectrum(np.array(df2["|dip|"]), time_step)

        # Get file labels without amino acid
        label1 = os.path.basename(file1).replace(amino_acid, '').strip('_')
        label2 = os.path.basename(file2).replace(amino_acid, '').strip('_')

        # Plot spectra with assigned colors and labels
        plot_spectra(freqs1, [spectrum1, spectrum2], f"Spectra for {amino_acid.upper()} pairs", 
                     colors=['black', 'red'], labels=[label1, label2])

    except Exception as e:
        print(f"Error processing pair ({file1}, {file2}): {e}")

def main():
    # Set the amino acid type to filter files ('trp' or 'ala')
    amino_acid = 'trp'  # Replace with 'ala' if needed

    # List all .dat files containing the amino acid name
    files = [os.path.join(root, name)
             for root, _, names in os.walk(os.getcwd())
             for name in names if name.endswith('.dat') and amino_acid in name]

    # Ensure pairs of files are processed together
    for i in range(0, len(files) - 1, 2):
        process_pair(files[i], files[i + 1], amino_acid)

if __name__ == '__main__':
    main()
