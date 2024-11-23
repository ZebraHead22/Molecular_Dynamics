import os
import re
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
    return autocorr[len(chunk) - 1:]

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

# Поиск основных пиков с учетом ширины на половине амплитуды по паре пиков на 1000 см
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



# Поиск основных пиков с учетом минимального расстояния между ними
# def find_main_peaks(xf_cm_inv_filtered, spectral_density_filtered, distance=200, max_peaks=5):
#     """
#     Находит пики с учетом минимального расстояния между ними:
#     - Минимальное расстояние между пиками: `distance` см⁻¹.
#     - Учитываются только `max_peaks` пиков на диапазон.

#     Параметры:
#     - xf_cm_inv_filtered: Массив частот.
#     - spectral_density_filtered: Массив спектральной плотности.
#     - distance: Минимальное расстояние между пиками в см⁻¹.
#     - max_peaks: Максимальное количество сохраняемых пиков.

#     Возвращает:
#     - peak_frequencies: Частоты пиков.
#     - peak_amplitudes: Амплитуды пиков.
#     - peak_widths_half_max: Ширины пиков на половине высоты.
#     """
#     ranges = [(i, i + 500) for i in range(0, 4000, 500)]
#     peak_frequencies = []
#     peak_amplitudes = []
#     peak_widths_half_max = []

#     for lower_bound, upper_bound in ranges:
#         # Фильтруем данные в текущем диапазоне
#         mask = (xf_cm_inv_filtered >= lower_bound) & (xf_cm_inv_filtered < upper_bound)
#         sub_xf = xf_cm_inv_filtered[mask]
#         sub_spectral_density = spectral_density_filtered[mask]

#         if len(sub_xf) > 0:
#             # Определяем минимальную высоту пика (10% от максимума в диапазоне)
#             max_amp = np.max(sub_spectral_density)
#             if max_amp <= 0:  # Если спектр в диапазоне нулевой или отрицательный
#                 continue
#             min_height = 0.1 * max_amp

#             # Ищем пики выше минимальной высоты
#             peaks, properties = find_peaks(sub_spectral_density, height=min_height, distance=distance)
#             if len(peaks) > 0:
#                 # Сортируем пики по высоте и оставляем не более max_peaks
#                 sorted_indices = np.argsort(properties["peak_heights"])[::-1]
#                 top_peaks = sorted_indices[:max_peaks]

#                 for peak_idx in top_peaks:
#                     peak = peaks[peak_idx]
#                     peak_freq = sub_xf[peak]
#                     peak_amp = sub_spectral_density[peak]

#                     # Расчет ширины на половине высоты
#                     results_half_max = peak_widths(sub_spectral_density, [peak], rel_height=0.5)
#                     width = results_half_max[0][0] * (sub_xf[1] - sub_xf[0])

#                     peak_frequencies.append(peak_freq)
#                     peak_amplitudes.append(peak_amp)
#                     peak_widths_half_max.append(width)

#     return peak_frequencies, peak_amplitudes, peak_widths_half_max


def annotate_peaks_with_overlap_handling(ax, frequencies, amplitudes, color, offset=0.02):
    """
    Аннотирует пики с учетом наложений аннотаций.

    Параметры:
    - ax: ось matplotlib для аннотаций.
    - frequencies: массив частот пиков.
    - amplitudes: массив амплитуд пиков.
    - color: цвет аннотаций и стрелок.
    - offset: вертикальное смещение аннотации при наложении.
    """
    annotations = []

    for freq, amp in zip(frequencies, amplitudes):
        # Проверяем, пересекается ли текущая аннотация с предыдущими
        overlap = False
        for prev_freq, prev_amp, prev_color in annotations:
            if abs(freq - prev_freq) < 50:  # Если частоты слишком близки
                overlap = True
                break

        # Если есть наложение, смещаем аннотацию вертикально
        if overlap:
            annotation_y = amp + offset * np.max(amplitudes)
        else:
            annotation_y = amp + 0.02 * np.max(amplitudes)

        # Добавляем стрелку и аннотацию
        ax.annotate(
            f"{freq:.1f}",
            xy=(freq, amp),
            xytext=(freq, annotation_y),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.5),
            fontsize=14,
            color=color,
            ha="center",
            rotation=45
        )
        # Сохраняем текущую аннотацию
        annotations.append((freq, annotation_y, color))


# Сохранение графиков: все на одной картинке для полного спектра и для каждого диапазона со старыми аннотациями
# def save_combined_graph_with_annotations(xf_cm_inv_filtered_list, spectral_density_filtered_list, titles, peak_data_list, prefix):
#     colors = ['black', 'red']  # Черный и красный цвета

#     # Полный спектр
#     plt.figure(figsize=(12, 8))
#     for i, (xf, sd, title) in enumerate(zip(xf_cm_inv_filtered_list, spectral_density_filtered_list, titles)):
#         plt.plot(xf, sd, label=title, color=colors[i])
#     plt.title(f"Full Spectra - {prefix}")
#     plt.xlabel("Frequency (cm^-1)")
#     plt.ylabel("Spectral Density (a.u.)")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{prefix}_full_spectra.png", dpi=300)
#     plt.close()
#     print(f"Saved full spectra graph: {prefix}_full_spectra.png")

#     # Диапазоны
#     ranges = [(start, start + 1000) for start in range(0, 6000, 1000)]
#     for start, end in ranges:
#         plt.figure(figsize=(12, 8))
#         for i, (xf, sd, title, peaks) in enumerate(zip(xf_cm_inv_filtered_list, spectral_density_filtered_list, titles, peak_data_list)):
#             mask = (xf >= start) & (xf < end)
#             if np.any(mask):  # Проверяем, есть ли данные в диапазоне
#                 plt.plot(xf[mask], sd[mask], label=title, color=colors[i])
#                 # Добавляем аннотации для пиков в пределах текущего диапазона
#                 for freq, amp in peaks:
#                     if start <= freq < end:  # Учитываем только пики в текущем диапазоне
#                         # Корректируем положение аннотации, чтобы не выходить за пределы области
#                         y_max = amp * 1.2
#                         text_y = min(amp * 1.1, y_max)
#                         plt.annotate(
#                             f'{freq:.2f}', xy=(freq, amp), xytext=(freq, text_y),
#                             textcoords='data', ha='center', va='bottom',
#                             rotation=45, fontsize=14, color=colors[i],
#                             arrowprops=dict(arrowstyle='->', color='green', lw=0.8)
#                         )
#         plt.title(f"Spectra Range {start}-{end} cm^-1 - {prefix}")
#         plt.xlabel("Frequency (cm^-1)")
#         plt.ylabel("Spectral Density (a.u.)")
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f"{prefix}_range_{start}_{end}.png", dpi=300)
#         plt.close()
#         print(f"Saved range graph: {prefix}_range_{start}_{end}.png")


def save_combined_graph_with_annotations(xf_cm_inv_filtered_list, spectral_density_filtered_list, titles, peak_data_list, prefix): # Это функция с учетом отдельной для аннотации. Закоменнтировать, если коментим аннотации
    colors = ['black', 'red']  # Черный и красный цвета

    # Полный спектр
    plt.figure(figsize=(12, 8))
    for i, (xf, sd, title) in enumerate(zip(xf_cm_inv_filtered_list, spectral_density_filtered_list, titles)):
        plt.plot(xf, sd, label=title, color=colors[i])
    plt.title(f"Full Spectra - {prefix}")
    plt.xlabel("Frequency (cm^-1)")
    plt.ylabel("Spectral Density (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_full_spectra.png", dpi=300)
    plt.close()
    print(f"Saved full spectra graph: {prefix}_full_spectra.png")

    # Диапазоны
    ranges = [(start, start + 1000) for start in range(0, 6000, 1000)]
    for start, end in ranges:
        plt.figure(figsize=(12, 8))
        for i, (xf, sd, title, peaks) in enumerate(zip(xf_cm_inv_filtered_list, spectral_density_filtered_list, titles, peak_data_list)):
            mask = (xf >= start) & (xf < end)
            if np.any(mask):  # Проверяем, есть ли данные в диапазоне
                plt.plot(xf[mask], sd[mask], label=title, color=colors[i])
                # Добавляем аннотации для пиков в пределах текущего диапазона
                peaks_in_range = [(freq, amp) for freq, amp in peaks if start <= freq < end]
                # Используем функцию для аннотирования пиков
                annotate_peaks_with_overlap_handling(plt.gca(), peaks_in_range, peaks_in_range, color=colors[i])

        plt.title(f"Spectra Range {start}-{end} cm^-1 - {prefix}")
        plt.xlabel("Frequency (cm^-1)")
        plt.ylabel("Spectral Density (a.u.)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{prefix}_range_{start}_{end}.png", dpi=300)
        plt.close()
        print(f"Saved range graph: {prefix}_range_{start}_{end}.png")


# Главная функция для обработки файлов
def process_files_in_directory_with_annotations(directory, output_file):
    xf_cm_inv_filtered_list = []
    spectral_density_filtered_list = []
    titles = []
    peak_data_list = []

    for address, dirs, names in os.walk(directory):
        for name in names:
            file_path = os.path.join(address, name)
            filename, file_extension = os.path.splitext(name)
            if file_extension == ".dat":
                print(f"-- Processing file: {filename}")

                df = pd.read_csv(file_path, delimiter=' ', index_col=None, header=[0])
                df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                                   'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
                df.dropna(how='all', axis=1, inplace=True)
                dip_magnitude = np.array(df['|dip|'].to_list())
                dip_magnitude -= np.mean(dip_magnitude)

                num_cores_to_use = 16
                dip_magnitude_corr = calculate_autocorrelation(dip_magnitude, num_cores=num_cores_to_use)
                window = hann(len(dip_magnitude_corr))
                dip_magnitude_windowed = dip_magnitude_corr * window

                time_step = 2e-15
                N = len(dip_magnitude_windowed)
                yf = fft(dip_magnitude_windowed)
                xf = fftfreq(N, time_step)[:N // 2]

                cutoff_frequency = 3e12
                cutoff_index = np.where(xf < cutoff_frequency)[0][-1] + 1
                yf[:cutoff_index] = 0

                xf_thz = xf * 1e-12
                xf_cm_inv = xf_thz / 0.03
                mask = xf_cm_inv <= 6000
                xf_cm_inv_filtered = xf_cm_inv[mask]
                spectral_density_filtered = 2.0 / N * np.abs(yf[:N // 2])[mask] * 10000

                # Поиск пиков и запись их в файл
                peak_frequencies, peak_amplitudes, peak_widths_half_max = find_main_peaks(
                    xf_cm_inv_filtered, spectral_density_filtered)
                for freq, amp, width in zip(peak_frequencies, peak_amplitudes, peak_widths_half_max):
                    output_file.write(f"AKF_{filename} -- {freq:.2f} -- {amp:.2f} -- {width:.2f}\n")

                # Добавляем данные для графиков
                xf_cm_inv_filtered_list.append(xf_cm_inv_filtered)
                spectral_density_filtered_list.append(spectral_density_filtered)
                titles.append(filename)
                peak_data_list.append(list(zip(peak_frequencies, peak_amplitudes)))

    # Построение комбинированных графиков с аннотациями
    save_combined_graph_with_annotations(
        xf_cm_inv_filtered_list, spectral_density_filtered_list, titles, peak_data_list, "AKF"
    )




if __name__ == '__main__':
    directory = os.getcwd()
    output_file_path = "peak_data.txt"

    with open(output_file_path, "w") as output_file:
        process_files_in_directory_with_annotations(directory, output_file)

    print("-- Processing complete.")

