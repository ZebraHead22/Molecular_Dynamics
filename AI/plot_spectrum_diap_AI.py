import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_widths
from scipy.signal.windows import hann
from multiprocessing import Pool, cpu_count

# Автокорреляция с разделением на участки
def autocorrelation_chunk(args):
    dip_magnitude, start, end = args
    chunk = dip_magnitude[start:end]
    autocorr = np.correlate(chunk, chunk, mode='full')
    return autocorr[len(chunk)-1:]

def calculate_autocorrelation(dip_magnitude, num_cores=None):
    N = len(dip_magnitude)
    if num_cores is None:
        num_cores = cpu_count()

    chunk_size = max(1000, N // (num_cores * 10))
    ranges = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

    with Pool(processes=num_cores) as pool:
        results = pool.map_async(autocorrelation_chunk, [(dip_magnitude, start, end) for start, end in ranges]).get()

    autocorr = np.zeros(len(dip_magnitude))
    for result in results:
        autocorr[:len(result)] += result

    return autocorr

# Генерация заголовка на основе имени файла
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

# Поиск основных пиков с учетом ширины на половине амплитуды
def find_main_peaks(xf_cm_inv_filtered, spectral_density_filtered):
    ranges = [(i, i + 500) for i in range(0, 4000, 500)]
    peak_frequencies = []
    peak_amplitudes = []
    peak_widths_half_max = []

    for lower_bound, upper_bound in ranges:
        mask = (xf_cm_inv_filtered >= lower_bound) & (xf_cm_inv_filtered < upper_bound)
        sub_xf = xf_cm_inv_filtered[mask]
        sub_spectral_density = spectral_density_filtered[mask]

        if len(sub_xf) > 0:
            peaks, _ = find_peaks(sub_spectral_density, height=0)
            if len(peaks) > 0:
                peak = peaks[np.argmax(sub_spectral_density[peaks])]
                peak_frequencies.append(sub_xf[peak])
                peak_amplitudes.append(sub_spectral_density[peak])

                # Расчет ширины на половине высоты
                results_half_max = peak_widths(sub_spectral_density, [peak], rel_height=0.5)
                width = results_half_max[0][0]
                peak_widths_half_max.append(width * (sub_xf[1] - sub_xf[0]))

    return peak_frequencies, peak_amplitudes, peak_widths_half_max

# Аннотация и сохранение данных пиков в файл
def annotate_and_save_peaks(output_file, prefix, filename, peak_frequencies, peak_amplitudes, peak_widths_half_max):
    for freq, amp, width in zip(peak_frequencies, peak_amplitudes, peak_widths_half_max):
        output_file.write(f"{prefix}_{filename} -- {freq:.2f} -- {amp:.2f} -- {width:.2f}\n")

# Функция для сохранения графиков по диапазонам
def save_spectrum_graph(xf_cm_inv_filtered, spectral_density_filtered, title, prefix, filename):
    # Разделение спектра на диапазоны с шагом 1000
    for start in range(0, 6000, 1000):
        end = start + 1000
        mask = (xf_cm_inv_filtered >= start) & (xf_cm_inv_filtered < end)
        sub_xf = xf_cm_inv_filtered[mask]
        sub_spectral_density = spectral_density_filtered[mask]

        # Строим график для каждого диапазона
        plt.figure(figsize=(8, 6))
        plt.plot(sub_xf, sub_spectral_density)
        plt.title(f"{title} - {prefix} - Range {start}-{end} cm^-1")
        plt.xlabel("Frequency (cm^-1)")
        plt.ylabel("Spectral Density (a.u.)")
        plt.grid(True)

        # Сохраняем график в файл
        plot_filename = f"{prefix}_{filename}_range_{start}_{end}.png"
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"Saved graph: {plot_filename}")

# Главная функция для обработки каждого файла
def process_file(name, output_file):
    filename, file_extension = os.path.splitext(name)
    if file_extension == ".dat":
        print(f"-- File {os.path.basename(filename)}")
        title = create_title(filename)
        print(f"-- Generated title: {title}")

        df = pd.read_csv(name, delimiter=' ', index_col=None, header=[0])
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                           'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        dip_magnitude = np.array(df['|dip|'].to_list())
        dip_magnitude -= np.mean(dip_magnitude)
        length = dip_magnitude.size
        print(f"-- Len of transient {length} points or {length * 2 / 1000000} ns")

        num_cores_to_use = 16
        print(f"-- Using {num_cores_to_use} cores")

        # === Spectrum with Autocorrelation ===
        dip_magnitude_corr = calculate_autocorrelation(dip_magnitude, num_cores=num_cores_to_use)
        window = hann(len(dip_magnitude_corr))
        dip_magnitude_windowed = dip_magnitude_corr * window

        time_step = 2e-15
        N = len(dip_magnitude_windowed)
        yf = fft(dip_magnitude_windowed)
        xf = fftfreq(N, time_step)[:N//2]

        cutoff_frequency = 3e12
        cutoff_index = np.where(xf < cutoff_frequency)[0][-1] + 1
        yf[:cutoff_index] = 0

        xf_thz = xf * 1e-12
        xf_cm_inv = xf_thz / 0.03
        mask = xf_cm_inv <= 6000
        xf_cm_inv_filtered = xf_cm_inv[mask]
        spectral_density_filtered = 2.0 / N * np.abs(yf[:N//2])[mask] * 10000

        peak_frequencies, peak_amplitudes, peak_widths_half_max = find_main_peaks(
            xf_cm_inv_filtered, spectral_density_filtered)

        # Запись данных в файл для спектра с автокорреляцией
        annotate_and_save_peaks(output_file, "AKF", os.path.basename(filename), peak_frequencies, peak_amplitudes, peak_widths_half_max)

        # Сохранение графиков для спектра с автокорреляцией
        save_spectrum_graph(xf_cm_inv_filtered, spectral_density_filtered, title, "AKF", os.path.basename(filename))

        # === Spectrum without Autocorrelation ===
        dip_magnitude_windowed_no_ac = dip_magnitude * window
        N_no_ac = len(dip_magnitude_windowed_no_ac)
        yf_no_ac = fft(dip_magnitude_windowed_no_ac)
        xf_no_ac = fftfreq(N_no_ac, time_step)[:N_no_ac//2]
        yf_no_ac[:cutoff_index] = 0

        xf_thz_no_ac = xf_no_ac * 1e-12
        xf_cm_inv_no_ac = xf_thz_no_ac / 0.03
        mask_no_ac = xf_cm_inv_no_ac <= 6000
        xf_cm_inv_filtered_no_ac = xf_cm_inv_no_ac[mask_no_ac]
        spectral_density_filtered_no_ac = 2.0 / N_no_ac * np.abs(yf_no_ac[:N_no_ac//2])[mask_no_ac] * 10000

        peak_frequencies, peak_amplitudes, peak_widths_half_max = find_main_peaks(
            xf_cm_inv_filtered_no_ac, spectral_density_filtered_no_ac)

        # Запись данных в файл для спектра без автокорреляции
        annotate_and_save_peaks(output_file, "no_AKF", os.path.basename(filename), peak_frequencies, peak_amplitudes, peak_widths_half_max)

        # Сохранение графиков для спектра без автокорреляции
        save_spectrum_graph(xf_cm_inv_filtered_no_ac, spectral_density_filtered_no_ac, title, "no_AKF", os.path.basename(filename))


if __name__ == '__main__':
    directory = os.getcwd()
    output_file = open("peak_data.txt", "w")
    for address, dirs, names in os.walk(directory):
        for name in names:
            process_file(os.path.join(address, name), output_file)

    output_file.close()
    print("-- Processing complete.")
