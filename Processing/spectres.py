import os
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

def find_main_peaks(xf_cm_inv_filtered, spectral_density_filtered):
    ranges = [(i, i + 200) for i in range(0, 6000, 200)]
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

def annotate_peaks(ax, peak_frequencies, peak_amplitudes):
    # Преобразуем списки в массивы NumPy, если это еще не массивы
    peak_frequencies = np.array(peak_frequencies)
    peak_amplitudes = np.array(peak_amplitudes)

    # Фильтруем пики по частотным диапазонам и выбираем самые высокие
    # Один пик из диапазона 0-1000
    low_range = np.where((peak_frequencies >= 0) & (peak_frequencies <= 1000))[0]
    top_low_peak = low_range[np.argmax(peak_amplitudes[low_range])] if len(low_range) > 0 else None

    # Один пик из диапазона 1000-2000
    mid_range = np.where((peak_frequencies > 1000) & (peak_frequencies <= 2000))[0]
    top_mid_peak = mid_range[np.argmax(peak_amplitudes[mid_range])] if len(mid_range) > 0 else None

    # Один пик из диапазона 3000-4000
    high_range = np.where((peak_frequencies > 3000) & (peak_frequencies <= 4000))[0]
    top_high_peak = high_range[np.argmax(peak_amplitudes[high_range])] if len(high_range) > 0 else None

    # Собираем все выбранные индексы пиков
    selected_indices = [idx for idx in [top_low_peak, top_mid_peak, top_high_peak] if idx is not None]

    # Находим максимальную амплитуду для дальнейших проверок
    max_amplitude = np.max(peak_amplitudes)

    # Аннотация выбранных пиков
    for idx in selected_indices:
        freq = peak_frequencies[idx]
        amp = peak_amplitudes[idx]

        # Проверяем наличие любых пиков в диапазоне ±200 от текущей частоты
        is_any_peak_near = any(
            abs(peak_frequencies[i] - freq) < 200 and i != idx for i in range(len(peak_frequencies))
        )

        if freq < 1000:
            # Если частота пика меньше 1000, переворачиваем аннотацию на 90 градусов и сдвигаем влево на 40
            ax.text(freq - 40, amp * 0.75, f'{freq:.2f}', fontsize=12, rotation=90, ha='right')
        elif is_any_peak_near:
            # Если есть пики в радиусе 200, размещаем аннотацию сверху
            ax.text(freq, amp * 1.1, f'{freq:.2f}', fontsize=12, ha='center')
        else:
            # Поднимаем аннотацию на amp * 0.1 для горизонтальных подписей
            ax.text(freq + 30, amp * 1.1, f'{freq:.2f}', fontsize=12)

def make_spectres():
    output_file = open("peak_data.txt", "w")
    files = os.listdir(os.getcwd())
    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            field_freq = re.search(r'\d{3,}', str(os.path.basename(filename)))
            if field_freq:
                field_freq = field_freq.group(0)
            else:
                field_freq = 'no_field'
            df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                      '0.0.1': 'Amplitude'}, inplace=True)
            df.insert(2, 'Amp×104', df['Amplitude']*(10**4))
            print(df.head())
            dfFreq = np.array(df['Frequency'].tolist())
            dfAmp = np.array(df['Amp×104'].tolist())
            # Ищем пики
            peak_frequencies, peak_amplitudes = find_main_peaks(
                dfFreq, dfAmp)
            # Запись в файл
            for freq, amp in zip(peak_frequencies, peak_amplitudes):
                output_file.write(
                    f"{os.path.basename(filename)} -- {freq:.2f} -- {amp:.2f}\n")
            # Строим графики
            plt.gcf().clear()
            fig, ax = plt.subplots()
            ax.plot(dfFreq, dfAmp, linewidth=1, c='black')  # Обычный график
            ax.grid()
            ax.set_xlabel('Frequency, $cm^{-1}$')
            ax.set_ylabel('Spectral Amplitude (a.u.)')
            # Аннотируем 4 самых высоких пика
            annotate_peaks(ax, peak_frequencies, peak_amplitudes)
            # plt.savefig(filename+'.png', dpi=600)
            # plt.savefig(filename+'.eps', format='eps')
            plt.show()

            
            '''
            Это дополнение, чтобы соотносить поле и искать примерно на нем точку максимума, а потом строить автоматические оконные графики со стрелками,
            указывающими на этот самый пик
            '''

            # # Ищем точку максимума на резонансной частоте
            # G = 50
            # max_amp_freq = df.loc[df['Amplitude'].where((df['Frequency'] < (int(
            #     field_freq) + G)) & (df['Frequency'] > (int(field_freq) - G))).idxmax(), 'Frequency']
            # max_amp = df.loc[df['Amplitude'].where((df['Frequency'] < (int(
            #     field_freq) + G)) & (df['Frequency'] > (int(field_freq) - G))).idxmax(), 'Amp×104']
            # # Ищем точку максимума без учета частоты поля
            # max_amp_freq = df.loc[df['Amplitude'].idxmax(), 'Frequency']
            # max_amp = df.loc[df['Amplitude'].idxmax(), 'Amp×104']
            # mess = 'Field ' + \
            #     str(field_freq) + " : " + str(max_amp_freq) + \
            #     ' - ' + str(max_amp) + '\n'
            # print(mess)

            # Оконные автоматические графики
            # D = 10  # Смещение для окон и подписей
            # plt.xlim(float(max_amp_freq) - D, float(max_amp_freq) + D)
            # plt.ylim(-max_amp*0.03, max_amp + max_amp*0.5)
            # plt.annotate(str(round(max_amp_freq, 2)), xy=(float(max_amp_freq), float(max_amp)), xytext=(float(
            #     max_amp_freq) + 0.5*D, float(max_amp) + float(max_amp)*0.05), arrowprops=dict(facecolor='red', shrink=0.05), fontsize=14)
            # plt.savefig(filename+'_window.png')


make_spectres()
