import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_widths
from scipy.signal.windows import hann, hamming
from multiprocessing import Pool, cpu_count
import argparse

# Автокорреляция с разделением на участки
def autocorrelation_chunk(args):
    """
    Compute the autocorrelation of a chunk of data.

    Parameters:
        args (tuple): A tuple containing the data array (dip_magnitude) and the start and end indices of the chunk.

    Returns:
        numpy.ndarray: The autocorrelation values for the chunk.
    """
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
    """
    Generate a title string based on the filename.

    Parameters:
        filename (str): The name of the file.

    Returns:
        str: A formatted title based on the filename.
    """
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
    """
    Annotate peak data and save it to a file.

    Parameters:
        output_file (file object): The file to write the peak data to.
        prefix (str): A prefix for the annotation.
        filename (str): The name of the file being processed.
        peak_frequencies (list): List of peak frequencies.
        peak_amplitudes (list): List of peak amplitudes.
        peak_widths_half_max (list): List of peak widths at half maximum.

    Returns:
        None
    """
    for freq, amp, width in zip(peak_frequencies, peak_amplitudes, peak_widths_half_max):
        try:
            output_file.write(f"{prefix}_{filename} -- {freq:.2f} -- {amp:.2f} -- {width:.2f}\n")
        except (ValueError, TypeError) as e:
            print(f"Error writing data for {filename}: {e}")

# Главная функция для обработки каждого файла
def process_file(name, output_file, num_cores):
    filename, file_extension = os.path.splitext(name)
    if file_extension == ".dat":
        print(f"-- File {os.path.basename(filename)}")
        title = create_title(filename)
        print(f"-- Generated title: {title}")

        df = pd.read_csv(name, delimiter=' ', index_col=None, header=[0])
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                           'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)

        df.insert(1, 'Time', df['frame']*2/1000)

        dip_magnitude = np.array(df['|dip|'].to_list())
        dip_magnitude -= np.mean(dip_magnitude)


        plt.figure()
        plt.plot(df['Time'].to_list(), dip_magnitude, color='black', linewidth=0.5)
        plt.title(title + ' Raw Data')
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (a.u.)")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{filename}_raw.png", dpi=300)


        length = dip_magnitude.size
        print(f"-- Len of transient {length} points or {length * 2 / 1000000} ns")

        print(f"-- Using {num_cores} cores")

        # === Spectrum with Autocorrelation ===
        dip_magnitude_corr = calculate_autocorrelation(dip_magnitude, num_cores=num_cores)
        
        # Построение графика автокорреляции
        plt.figure()
        plt.plot(df['Time'].to_list(), dip_magnitude_corr, color='black', linewidth=0.5)
        plt.title(title + " Autocorrelation")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (a.u.)")
        plt.tight_layout()
        # plt.xlim([0, 100])
        # plt.ylim([-2000, 2000])
        plt.grid(True)
        plt.savefig(f"{filename}_autocorrelation.png", dpi=300)

        window = hann(len(dip_magnitude_corr))
        dip_magnitude_windowed = dip_magnitude_corr * window

        #Построение графика наложения окна Ханна
        plt.figure()
        plt.plot(df['Time'].to_list(), dip_magnitude_windowed, color='black', linewidth=0.5)
        plt.title("Hann Window Applied")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (a.u.)")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{filename}_hann_window.png", dpi=300)

        time_step = 2e-15
        N = len(dip_magnitude_windowed) #_windowed if Hann
        yf = fft(dip_magnitude_windowed) # too
        xf = fftfreq(N, time_step)[:N//2]

        cutoff_frequency = 3e12
        cutoff_index = np.where(xf < cutoff_frequency)[0][-1] + 1
        yf[:cutoff_index] = 0

        xf_thz = xf * 1e-12
        xf_cm_inv = xf_thz / 0.03
        mask = xf_cm_inv <= 4000
        xf_cm_inv_filtered = xf_cm_inv[mask]
        spectral_density_filtered = 2.0 / N * np.abs(yf[:N//2])[mask]

        peak_frequencies, peak_amplitudes, peak_widths_half_max = find_main_peaks(
            xf_cm_inv_filtered, spectral_density_filtered)

        # Построение спектра
        plt.figure()
        plt.plot(xf_cm_inv_filtered, spectral_density_filtered, color='black', linewidth=0.5, label="Spectral ACF EDM")
        for freq in peak_frequencies:
            plt.axvline(x=freq, color='red', linewidth=0.5, linestyle='--', label=f"Peak {freq:.1f} cm$^{-1}$")
        plt.title(title)
        plt.xlabel("Frequency (cm$^{-1}$)")
        plt.ylabel("Spectral ACF EDM Amplitude (a.u.)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{filename}_spectrum.png", dpi=300)

        # Запись данных в файл для спектра с автокорреляцией
        annotate_and_save_peaks(output_file, "AKF", os.path.basename(filename), peak_frequencies, peak_amplitudes, peak_widths_half_max)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process dipole moment data files and generate autocorrelation and spectral density plots.")
    parser.add_argument('-d', '--directory', type=str, default=os.getcwd(), help="Directory to process (default: current working directory)")
    parser.add_argument('-o', '--output', type=str, default="peak_data.txt", help="Output file for peak data (default: peak_data.txt)")
    parser.add_argument('-c', '--cores', type=int, default=cpu_count(), help="Number of CPU cores to use (default: all available cores)")

    args = parser.parse_args()
    directory = args.directory
    output_filename = args.output
    num_cores = args.cores

    with open(output_filename, "w") as output_file:
        for address, dirs, names in os.walk(directory):
            for name in names:
                process_file(os.path.join(address, name), output_file, num_cores)

    print("-- Processing complete.")
