import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hann, correlate
from multiprocessing import Pool, cpu_count


def create_title(filename):
    match = re.search(r'(\w+)_(\d+)_(\w+)', filename)
    if match:
        amino_acid = match.group(1).upper()
        count = match.group(2)
        chain_type = match.group(3).capitalize()
        return f"{amino_acid} {chain_type} N={count}"
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


def find_peaks(frequencies, spectrum, f_min, f_max, step=200):
    """Находит 1 максимальный пик в каждом диапазоне с шагом step."""
    peaks = []
    for start in range(f_min, f_max, step):
        mask = (frequencies >= start) & (frequencies < start + step)
        if not np.any(mask):
            continue

        local_freqs = frequencies[mask]
        local_spectrum = spectrum[mask]

        if len(local_freqs) > 0:
            max_idx = np.argmax(local_spectrum)
            peak_freq = local_freqs[max_idx]
            peak_value = local_spectrum[max_idx]

            # Определение ширины пика
            half_max = peak_value / 2
            left_idx = np.where(local_spectrum[:max_idx] < half_max)[0]
            right_idx = np.where(local_spectrum[max_idx:] < half_max)[0]

            left_freq = local_freqs[left_idx[-1]] if left_idx.size > 0 else peak_freq
            right_freq = local_freqs[max_idx + right_idx[0]] if right_idx.size > 0 else peak_freq

            width = right_freq - left_freq
            peaks.append((peak_freq, peak_value, width))

    return sorted(peaks, key=lambda x: x[0])


def save_peaks_to_file(output_dir, freq_range, peaks, label, recorded_ranges):
    """Сохраняет данные о пиках в файл, избегая дублирования."""
    txt_file = os.path.join(output_dir, f"{label}_peaks.txt")
    range_key = (freq_range[0], freq_range[1], label)
    
    if range_key in recorded_ranges:
        return  # Если диапазон уже записан, пропускаем

    recorded_ranges.add(range_key)  # Добавляем диапазон в записанные

    with open(txt_file, 'a') as file:
        file.write(f"Range: {freq_range[0]}-{freq_range[1]} cm⁻¹\n")
        for peak_freq, peak_value, width in peaks:
            file.write(f"Frequency: {peak_freq:.3f} cm⁻¹, Width: {width:.3f}\n")
        file.write("\n")

def plot_spectra(positive_freqs, spectra, title, colors, labels, output_dir, recorded_ranges):
    freq_ranges = [(10, 1000), (1000, 2000), (2000, 3000),
                (3000, 4000), (4000, 5000), (5000, 6000)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f_min, f_max in freq_ranges:
        plt.figure(figsize=(10, 8))

        for spectrum, color, label in zip(spectra, colors, labels):
            mask = (positive_freqs >= f_min) & (positive_freqs < f_max)
            plot_freqs = positive_freqs[mask]
            plot_spectrum = spectrum[mask]

            plt.plot(plot_freqs, plot_spectrum, color=color, label=label, alpha=0.6 if color == 'black' else 1.0)

            peaks = find_peaks(plot_freqs, plot_spectrum, f_min, f_max)
            for peak_freq, peak_value, width in peaks:
                plt.text(peak_freq, peak_value, f"{peak_freq:.1f}", color=color, fontsize=11, rotation=45)

            save_peaks_to_file(output_dir, (f_min, f_max), peaks, label, recorded_ranges)

        plt.xlabel("Frequency (cm⁻¹)")
        plt.ylabel("Spectral ACF EDM Amplitude (a.u.)")
        plt.title(f"{title} - {f_min}-{f_max} cm⁻¹")
        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(output_dir, f"{f_min}-{f_max}_spectrum.png"), dpi=300)
        plt.close()



def generate_output_dir(file1, file2):
    """Генерация имени выходной папки без дубликатов."""
    common_parts = []
    unique_parts = []

    split1 = file1.split('_')
    split2 = file2.split('_')

    for part1, part2 in zip(split1, split2):
        if part1 == part2:
            common_parts.append(part1)
        else:
            unique_parts.extend(sorted([part1, part2]))

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

        recorded_ranges = set()  # Хранит уже записанные диапазоны

        plot_spectra(freqs1, [spectrum1, spectrum2], f"Spectra for {label1} & {label2}",
            colors=['red', 'black'], labels=[label1, label2], output_dir=output_dir, recorded_ranges=recorded_ranges)


    except Exception as e:
        print(f"Error processing pair ({file1}, {file2}): {e}")


def main():
    files = [os.path.join(root, name)
             for root, _, names in os.walk(os.getcwd())
             for name in names if name.endswith('.dat')]

    pairs = set()  # Используем множество, чтобы исключить дубликаты
    for i, file1 in enumerate(files):
        for j in range(i + 1, len(files)):
            file2 = files[j]

            # Разделяем имя файла на компоненты
            match1 = re.match(r'(\w+)_(\d+)_(\w+)\.dat', os.path.basename(file1))
            match2 = re.match(r'(\w+)_(\d+)_(\w+)\.dat', os.path.basename(file2))
            if not match1 or not match2:
                continue

            amino_acid1, count1, chain_type1 = match1.groups()
            amino_acid2, count2, chain_type2 = match2.groups()

            # Проверяем, что аминокислоты одинаковые
            if amino_acid1 != amino_acid2:
                continue

            # Условия для формирования пар
            if count1 == count2 and chain_type1 != chain_type2:
                pairs.add(tuple(sorted([file1, file2])))
            elif chain_type1 == chain_type2 and count1 != count2:
                pairs.add(tuple(sorted([file1, file2])))

    with Pool(cpu_count()) as pool:
        pool.map(process_pair, list(pairs))


if __name__ == "__main__":
    main()