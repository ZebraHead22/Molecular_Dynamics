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
    # Находим индексы 4 самых высоких пиков
    top_peaks_indices = np.argsort(peak_amplitudes)[-4:]  # Индексы 4 самых больших пиков
    top_peaks_indices = sorted(top_peaks_indices)  # Сортируем по частоте для правильного порядка

    for idx in top_peaks_indices:
        freq = peak_frequencies[idx]
        amp = peak_amplitudes[idx]

        # Проверяем наличие более "правого" пика
        is_right_overlap = any(
            peak_frequencies[i] < freq + 1000 and i != idx for i in top_peaks_indices
        )
        # Проверяем наличие пика "слева"
        is_left_overlap = any(
            peak_frequencies[i] > freq - 1000 and peak_frequencies[i] < freq and i != idx for i in top_peaks_indices
        )

        # Логика смещения аннотаций: если справа перекрытие — пытаемся сдвинуть влево,
        # но если слева тоже есть пик, тогда оставляем подпись справа.
        if is_right_overlap and not is_left_overlap:
            # Сдвигаем влево, если справа есть пик, а слева нет
            x_pos = freq - 1000
        else:
            # Оставляем справа, если перекрытие слева или нет перекрытий справа
            x_pos = freq + 30

        # Сдвигаем текст по амплитуде вниз
        ax.text(x_pos, amp * 0.95, f'{freq:.2f}', 
                fontsize=12, fontname='Courier New', weight='bold')


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
            plt.savefig(filename+'.png', dpi=600)
            plt.savefig(filename+'.eps', format='eps')
            # plt.show()



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
