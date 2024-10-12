import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal.windows import hann
from multiprocessing import Pool, cpu_count

from scipy.signal import find_peaks

def annotate_peaks(ax, peak_frequencies, peak_amplitudes, xlim_max=6000):
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

    # Получим границы оси Y для проверки выхода за пределы
    ylim_lower, ylim_upper = ax.get_ylim()

    # Функция для аннотации
    def add_annotation(ax, freq, amp, label, offset_y, offset_x=0, draw_arrow=False):
        text_y = amp + offset_y
        if text_y > ylim_upper:
            text_y = ylim_upper - (ylim_upper * 0.05)
        if freq + offset_x > xlim_max:
            offset_x = xlim_max - freq  # Сдвигаем аннотацию внутрь оси X

        ax.text(freq + offset_x, text_y, label, fontsize=12, rotation=0, ha='center')

        # Добавляем стрелку при необходимости
        if draw_arrow:
            ax.annotate('', xy=(freq, amp), xytext=(freq + offset_x, text_y),
                        arrowprops=dict(facecolor='none', edgecolor='red', linestyle='dashed', 
                                        lw=0.5, shrink=0.05))

    annotations = []  # Список для хранения аннотаций пиков
    legend_entries = []  # Список для записи в легенду
    peak_counter = 1  # Счётчик для нумерации пиков

    for (low, high), num_peaks in ranges.items():
        # Найдем пики в текущем диапазоне с более низким порогом для поиска
        current_range = np.where((peak_frequencies > low) & (peak_frequencies <= high))[0]
        if len(current_range) == 0:
            print(f"Диапазон {low}-{high} см⁻¹: нет доступных пиков")
            continue

        # Установим порог для поиска менее значимых пиков, используя функцию find_peaks
        selected_peaks, _ = find_peaks(peak_amplitudes[current_range], height=0.05 * max_peak_amplitude)
        
        # Если не хватает пиков, выводим предупреждение
        if len(selected_peaks) < num_peaks:
            print(f"В диапазоне {low}-{high} см⁻¹ недостаточно пиков. Найдено: {len(selected_peaks)}.")

        # Выбираем нужное количество пиков
        selected_indices = current_range[selected_peaks[:num_peaks]]

        # Отладочная информация
        print(f"Диапазон {low}-{high} см⁻¹: найдено {len(selected_indices)} пиков")

        # Добавляем аннотацию для каждого выбранного пика
        for idx in selected_indices:
            freq = peak_frequencies[idx]
            amp = peak_amplitudes[idx]
            add_annotation(ax, freq, amp, f'{peak_counter}', offset_y=max_peak_amplitude * 0.06)

            # Добавляем запись в легенду
            legend_entries.append(f'{peak_counter}: {freq:.2f} см⁻¹')
            peak_counter += 1

    # Добавляем легенду на график
    legend_text = '\n'.join(legend_entries)
    ax.text(0.95, 0.95, legend_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.6))

    return ax
