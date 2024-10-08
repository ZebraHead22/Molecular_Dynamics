import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal.windows import hann
from multiprocessing import Pool, cpu_count

'''
Тут строим пары изображений, без АКФ и с ней. В тайтле пишем, что есть что.
Ищем пики кажде 500 см, созраняем в .eps и .png.
'''

# Function for autocorrelation


def autocorrelation_chunk(args):
    dip_magnitude, start, end = args
    chunk = dip_magnitude[start:end]
    autocorr = np.correlate(chunk, chunk, mode='full')
    return autocorr[len(chunk)-1:]


def calculate_autocorrelation(dip_magnitude, num_cores=None):
    N = len(dip_magnitude)
    if num_cores is None:
        num_cores = cpu_count()  # Use all available cores if not specified

    # Minimum 1000 elements per chunk
    chunk_size = max(1000, N // (num_cores * 10))
    ranges = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

    with Pool(processes=num_cores) as pool:
        results = pool.map_async(autocorrelation_chunk, [(
            dip_magnitude, start, end) for start, end in ranges]).get()

    autocorr = np.zeros(len(dip_magnitude))
    for result in results:
        autocorr[:len(result)] += result

    return autocorr


def create_title(filename):
    # Extract the number and other components from the filename
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


def find_main_peaks(xf_cm_inv_filtered, spectral_density_filtered):
    ranges = [(i, i + 500) for i in range(0, 6000, 500)]
    peak_frequencies = []
    peak_amplitudes = []

    for lower_bound, upper_bound in ranges:
        mask = (xf_cm_inv_filtered >= lower_bound) & (
            xf_cm_inv_filtered < upper_bound)
        sub_xf = xf_cm_inv_filtered[mask]
        sub_spectral_density = spectral_density_filtered[mask]

        if len(sub_xf) > 0:
            peaks, _ = find_peaks(sub_spectral_density, height=0)
            if len(peaks) > 0:
                peak = peaks[np.argmax(sub_spectral_density[peaks])]
                peak_frequencies.append(sub_xf[peak])
                peak_amplitudes.append(sub_spectral_density[peak])

    return peak_frequencies, peak_amplitudes


if __name__ == '__main__':
    directory = os.getcwd()
    output_file = open("peak_data.txt", "w")
    for address, dirs, names in os.walk(directory):
        for name in names:
            filename, file_extension = os.path.splitext(name)
            if file_extension == ".dat":
                print(f"-- File {os.path.basename(filename)}")

                start_time = time.time()  # Start timing

                # Generate title based on filename
                title = create_title(filename)
                print(f"-- Generated title: {title}")

                df = pd.read_csv(
                    address + "/" + name, delimiter=' ', index_col=None, header=[0])
                df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                                   'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
                df.dropna(how='all', axis=1, inplace=True)
                dip_magnitude = np.array(df['|dip|'].to_list())
                dip_magnitude = dip_magnitude - np.mean(dip_magnitude)
                length = dip_magnitude.size
                formatted_length = f"{length:,}".replace(",", ".")
                print(
                    f"-- Len of transient {formatted_length} points or {length * 2 / 1000000} ns")

                num_cores_to_use = 16  # Specify the number of cores to use
                total_cores = cpu_count()
                print(f"-- Available CPU cores: {total_cores}")
                print(
                    f"-- Number of cores used for the task: {num_cores_to_use}")

                # === Spectrum with Autocorrelation ===
                dip_magnitude_corr = calculate_autocorrelation(
                    dip_magnitude, num_cores=num_cores_to_use)
                window = hann(len(dip_magnitude_corr))
                dip_magnitude_windowed = dip_magnitude_corr * window

                time_step = 2e-15
                N = len(dip_magnitude_windowed)
                yf = fft(dip_magnitude_windowed)
                xf = fftfreq(N, time_step)[:N//2]

                cutoff_frequency = 3e12  # Cutoff frequency in Hz (1 THz)
                cutoff_index = np.where(xf < cutoff_frequency)[0][-1] + 1
                yf[:cutoff_index] = 0  # Zeroing out low-frequency components

                xf_thz = xf * 1e-12
                xf_cm_inv = xf_thz / 0.03
                mask = xf_cm_inv <= 6000
                xf_cm_inv_filtered = xf_cm_inv[mask]
                spectral_density_filtered = 2.0/N * np.abs(yf[:N//2])[mask]
                spectral_density_filtered = np.array(
                    [x * 10000 for x in spectral_density_filtered])

                peak_frequencies, peak_amplitudes = find_main_peaks(
                    xf_cm_inv_filtered, spectral_density_filtered)

                # Запись данных в файл
                for freq, amp in zip(peak_frequencies, peak_amplitudes):
                    output_file.write(
                        f"{filename} -- {freq:.2f} -- {amp:.2f}\n")

                plt.gcf().clear()
                plt.plot(xf_cm_inv_filtered,
                         spectral_density_filtered, c='black')
                plt.xlim(0, 6000)
                plt.xlabel('Frequency ($cm^{-1}$)')
                plt.ylabel('Spectral Amplitude (a.u.)')
                plt.title(title + ' (With Autocorrelation)')
                plt.grid()
                plt.savefig(filename + '_ac.eps', format='eps')
                plt.savefig(filename + '_ac.png', dpi=600)

                # === Spectrum without Autocorrelation ===
                dip_magnitude_windowed_no_ac = dip_magnitude * window

                N_no_ac = len(dip_magnitude_windowed_no_ac)
                yf_no_ac = fft(dip_magnitude_windowed_no_ac)
                xf_no_ac = fftfreq(N_no_ac, time_step)[:N_no_ac//2]

                # Zeroing out low-frequency components
                yf_no_ac[:cutoff_index] = 0

                xf_thz_no_ac = xf_no_ac * 1e-12
                xf_cm_inv_no_ac = xf_thz_no_ac / 0.03
                mask_no_ac = xf_cm_inv_no_ac <= 6000
                xf_cm_inv_filtered_no_ac = xf_cm_inv_no_ac[mask_no_ac]
                spectral_density_filtered_no_ac = 2.0/N_no_ac * \
                    np.abs(yf_no_ac[:N_no_ac//2])[mask_no_ac]
                spectral_density_filtered_no_ac = np.array(
                    [x * 10000 for x in spectral_density_filtered_no_ac])

                peak_frequencies, peak_amplitudes = find_main_peaks(
                    xf_cm_inv_filtered, spectral_density_filtered)

                # Запись данных в файл
                for freq, amp in zip(peak_frequencies, peak_amplitudes):
                    output_file.write(
                        f"AKF {filename} -- {freq:.2f} -- {amp:.2f}\n")

                plt.gcf().clear()
                plt.plot(xf_cm_inv_filtered_no_ac,
                         spectral_density_filtered_no_ac, c='black')
                plt.xlim(0, 6000)
                plt.xlabel('Frequency ($cm^{-1}$)')
                plt.ylabel('Spectral Amplitude (a.u.)')
                plt.title(title + ' (Without Autocorrelation)')
                plt.grid()
                # plt.savefig(filename + '_no_ac.png', dpi=600)
                plt.savefig(filename + '_no_ac.eps', format='eps')
                plt.savefig(filename + '_no_ac.png', dpi = 600)

                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time
                formatted_time = time.strftime(
                    "%H:%M:%S", time.gmtime(elapsed_time))
                print(
                    f"-- Processing time for {os.path.basename(filename)}: {formatted_time} \n")
