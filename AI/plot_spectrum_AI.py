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

# def annotate_peaks(ax, peak_frequencies, peak_amplitudes, thresholds=[1000, 2000, 3000, 4000], tolerance=100):
#     # Преобразуем списки в массивы NumPy, если это еще не массивы
#     peak_frequencies = np.array(peak_frequencies)
#     peak_amplitudes = np.array(peak_amplitudes)

#     # Функция для аннотации и соединения стрелочкой
#     def add_annotation_with_arrow(ax, x, y, text, x_text, y_text):
#         ax.annotate(text, xy=(x, y), xytext=(x_text, y_text),
#                     arrowprops=dict(facecolor='red', shrink=0.05),
#                     fontsize=12, ha='center')

#     # Функция для добавления аннотаций, включая все правила
#     def add_annotation(ax, freq, amp, label, prev_freq=None, prev_amp=None, was_replaced=False):
#         if freq < 200:
#             # Если частота пика меньше 200, повернуть надпись на 90 градусов и сдвинуть вправо на 30
#             ax.text(freq + 30, amp * 1.1, f'{freq:.2f}', fontsize=12, rotation=90, ha='left')
#             return

#         if prev_freq and was_replaced:
#             # Если был заменен пик, аннотацию к старому пику переворачиваем
#             ax.text(prev_freq, prev_amp * 1.2, f'{prev_freq:.2f}', fontsize=12, rotation=90, ha='center')

#         if freq < 1000:
#             # Если частота пика меньше 1000, переворачиваем аннотацию на 90 градусов и сдвигаем влево на 40
#             ax.text(freq - 40, amp * 0.75, f'{freq:.2f}', fontsize=12, rotation=90, ha='right')
#         else:
#             # Основное правило аннотации для пиков больше 1000
#             ax.text(freq + 30, amp * 1.1, f'{freq:.2f}', fontsize=12)

#     for i, threshold in enumerate(thresholds):
#         # Устанавливаем диапазон поиска для текущего порога
#         lower_bound = 0 if i == 0 else thresholds[i - 1]
#         upper_bound = threshold
#         # Находим 4 пика в диапазоне
#         current_range = np.where((peak_frequencies > lower_bound) & (peak_frequencies <= upper_bound))[0]
#         if len(current_range) == 0:
#             continue

#         # Находим 4 самых высоких пика в текущем диапазоне
#         top_4_peaks = np.argsort(peak_amplitudes[current_range])[-4:]
#         selected_indices = current_range[top_4_peaks]

#         # Из 4 пиков выбираем максимальный для аннотации
#         max_peak_idx = selected_indices[np.argmax(peak_amplitudes[selected_indices])]
#         freq = peak_frequencies[max_peak_idx]
#         amp = peak_amplitudes[max_peak_idx]

#         # Проверяем, есть ли пик с большей амплитудой в предыдущем диапазоне, если мы находимся близко к порогу
#         if lower_bound < freq < upper_bound and (upper_bound - freq) < tolerance:
#             prev_range = np.where((peak_frequencies > lower_bound - tolerance) & 
#                                   (peak_frequencies < upper_bound + tolerance))[0]
#             if len(prev_range) > 0:
#                 max_prev_idx = prev_range[np.argmax(peak_amplitudes[prev_range])]
#                 if peak_amplitudes[max_prev_idx] > amp:
#                     # Заменяем на новый пик и аннотируем оба
#                     add_annotation(ax, peak_frequencies[max_prev_idx], peak_amplitudes[max_prev_idx], 
#                                    f'{peak_frequencies[max_prev_idx]:.2f}', prev_freq=freq, prev_amp=amp, was_replaced=True)
#                     continue  # Пропускаем добавление основной аннотации, т.к. она уже сделана

#         # Аннотация для максимального пика в текущем диапазоне
#         add_annotation(ax, freq, amp, f'{freq:.2f}')

def annotate_peaks(ax, peak_frequencies, peak_amplitudes):
    # Преобразуем списки в массивы NumPy, если это еще не массивы
    peak_frequencies = np.array(peak_frequencies)
    peak_amplitudes = np.array(peak_amplitudes)
    
    # Словарь с диапазонами и количеством пиков для поиска
    ranges = {
        (0, 500): 1,
        (500, 1000): 1,
        (1000, 2000): 2,
        (2500, 3500): 3
    }

    # Найдем максимальный пик для аннотации
    max_peak_amplitude = np.max(peak_amplitudes)

    # Функция для аннотации
    def add_annotation(ax, freq, amp, label, offset_y, offset_x=0):
        ax.text(freq + offset_x, amp + offset_y, label, fontsize=12, 
                rotation=90, ha='center')

    annotations = []  # Хранение данных для аннотаций
    for (low, high), num_peaks in ranges.items():
        # Найдем пики в текущем диапазоне
        current_range = np.where((peak_frequencies > low) & (peak_frequencies <= high))[0]
        if len(current_range) == 0:
            continue

        # Находим нужное количество максимальных пиков
        top_peaks = np.argsort(peak_amplitudes[current_range])[-num_peaks:]
        selected_indices = current_range[top_peaks]

        # Добавляем данные для аннотаций
        for idx in selected_indices:
            freq = peak_frequencies[idx]
            amp = peak_amplitudes[idx]
            annotations.append((freq, amp))

    # Сортируем аннотации по частоте для последовательного отображения
    annotations = sorted(annotations, key=lambda x: x[0])

    # Обрабатываем аннотации
    for i, (freq, amp) in enumerate(annotations):
        offset_y = max_peak_amplitude * 0.06  # Отступ для аннотации над пиком
        offset_x = 0

        # Проверка для частот пиков меньше 200
        if freq < 200:
            add_annotation(ax, freq, amp, f'{freq:.2f}', offset_y, offset_x=30)
            continue

        # Проверка для частот пиков меньше 1000
        if freq < 1000:
            add_annotation(ax, freq, amp, f'{freq:.2f}', offset_y, offset_x=-40)
            continue

        # Проверяем, есть ли пики, которые расположены близко друг к другу (< 350)
        if i > 0 and (freq - annotations[i - 1][0]) < 350:
            # Поднимаем аннотации на amp_max * 0.26
            offset_y = max_peak_amplitude * 0.26
            # Для пика с меньшей частотой смещаем аннотацию влево
            if freq < annotations[i - 1][0]:
                offset_x = -40
            # Для пика с большей частотой смещаем аннотацию вправо
            else:
                offset_x = 40

        # Обычная аннотация
        add_annotation(ax, freq, amp, f'{freq:.2f}', offset_y)

    return ax




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
                fig, ax = plt.subplots()
                ax.plot(xf_cm_inv_filtered,
                         spectral_density_filtered, c='black')
                ax.grid()
                ax.set_xlim(0, 6000)
                ax.set_xlabel('Frequency ($cm^{-1}$)')
                ax.set_ylabel('Spectral Amplitude (a.u.)')
                ax.set_title(title + ' (With Autocorrelation)')
                annotate_peaks(ax, peak_frequencies, peak_amplitudes)
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
                    xf_cm_inv_filtered_no_ac, spectral_density_filtered_no_ac)

                # Запись данных в файл
                for freq, amp in zip(peak_frequencies, peak_amplitudes):
                    output_file.write(
                        f"AKF {filename} -- {freq:.2f} -- {amp:.2f}\n")

                plt.gcf().clear()
                fig, ax = plt.subplots()
                ax.plot(xf_cm_inv_filtered_no_ac,
                         spectral_density_filtered_no_ac, c='black')
                ax.grid()
                ax.set_xlim(0, 6000)
                ax.set_xlabel('Frequency ($cm^{-1}$)')
                ax.set_ylabel('Spectral Amplitude (a.u.)')
                ax.set_title(title + ' (Without Autocorrelation)')
                annotate_peaks(ax, peak_frequencies, peak_amplitudes)
                plt.savefig(filename + '_no_ac.eps', format='eps')
                plt.savefig(filename + '_no_ac.png', dpi=600)

                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time
                formatted_time = time.strftime(
                    "%H:%M:%S", time.gmtime(elapsed_time))
                print(
                    f"-- Processing time for {os.path.basename(filename)}: {formatted_time} \n")
