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
    autocorr = correlate(dipole_moment, dipole_moment, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    hann_window = hann(len(autocorr))
    autocorr_windowed = autocorr * hann_window

    freq = fftfreq(len(autocorr_windowed), d=time_step)
    spectrum = np.abs(fft(autocorr_windowed))

    positive_freqs = freq[freq >= 0]
    positive_spectrum = spectrum[freq >= 0]

    cutoff_index = np.searchsorted(positive_freqs, 10)
    positive_freqs = positive_freqs[cutoff_index:]
    positive_spectrum = positive_spectrum[cutoff_index:]

    return positive_freqs, positive_spectrum


def plot_spectra(positive_freqs, spectra, title, colors, labels, output_dir):
    freq_ranges = [(0, 1000), (1000, 2000), (2000, 3000),
                   (3000, 4000), (4000, 5000), (5000, 6000)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (f_min, f_max) in enumerate(freq_ranges):
        plt.figure(figsize=(8, 6))
        for spectrum, color, label in zip(spectra, colors, labels):
            mask = (positive_freqs >= f_min) & (positive_freqs < f_max)
            plot_freqs = positive_freqs[mask]
            plot_spectrum = spectrum[mask]
            alpha = 0.4 if color == 'red' else 1.0  # Прозрачность для красного
            plt.plot(plot_freqs, plot_spectrum, color=color, alpha=alpha, label=label)

        plt.xlabel("Frequency (cm⁻¹)")
        plt.ylabel("Spectral ACF EDM Amplitude (a.u.)")
        plt.title(f"{title} - {f_min}-{f_max} cm⁻¹")
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(output_dir, f"{f_min}-{f_max}_spectrum.png"), dpi=300)
        plt.close()



def generate_output_dir(file1, file2):
    common_parts = []
    unique_parts = []

    split1 = file1.split('_')
    split2 = file2.split('_')

    for part1, part2 in zip(split1, split2):
        if part1 == part2:
            common_parts.append(part1)
        else:
            unique_parts.extend([part1, part2])

    output_dir_name = '_'.join(common_parts + unique_parts)
    return output_dir_name


def process_pair(args):
    file1, file2 = args
    try:
        df1 = pd.read_csv(file1, sep=' ')
        df2 = pd.read_csv(file2, sep=' ')

        df1.dropna(how='all', axis=1, inplace=True)
        df1.rename(columns={'#': 'frame', 'Unnamed: 8': '|dip|'}, inplace=True)

        df2.dropna(how='all', axis=1, inplace=True)
        df2.rename(columns={'#': 'frame', 'Unnamed: 8': '|dip|'}, inplace=True)

        time_step = 2e-6

        freqs1, spectrum1 = compute_spectrum(np.array(df1["|dip|"]), time_step)
        freqs2, spectrum2 = compute_spectrum(np.array(df2["|dip|"]), time_step)

        label1 = os.path.basename(file1).replace('.dat', '')
        label2 = os.path.basename(file2).replace('.dat', '')

        output_dir = generate_output_dir(label1, label2)
        plot_spectra(freqs1, [spectrum1, spectrum2], f"Spectra for {label1} & {label2}",
                     colors=['black', 'red'], labels=[label1, label2], output_dir=output_dir)

    except Exception as e:
        print(f"Error processing pair ({file1}, {file2}): {e}")


def main():
    files = [os.path.join(root, name)
             for root, _, names in os.walk(os.getcwd())
             for name in names if name.endswith('.dat')]

    pairs = []
    for i, file1 in enumerate(files):
        for j in range(i + 1, len(files)):
            file2 = files[j]
            amino_acid1, count1, chain_type1 = file1.split('_')[:3]
            amino_acid2, count2, chain_type2 = file2.split('_')[:3]

            if amino_acid1 != amino_acid2:
                continue  # Пропустить, если аминокислоты разные
            if count1 != count2 and chain_type1 != chain_type2:
                continue  # Пропустить, если и количество, и цепь различны одновременно

            pairs.append((file1, file2))

    num_processes = min(16, cpu_count())
    with Pool(processes=num_processes) as pool:
        pool.map(process_pair, pairs)


if __name__ == '__main__':
    main()
