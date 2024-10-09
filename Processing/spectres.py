# Здесь строим кучу спектров из набора дат файлов в одной папке
import os
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


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
            plt.plot(dfFreq, dfAmp, linewidth=1, c='black')  # Обычный графики
            plt.grid()
            plt.xlabel('Frequency, $cm^{-1}$')
            plt.ylabel('Spectral Amplitude (a.u.)')
            # plt.savefig(filename+'.png', dpi=600)
            # plt.savefig(filename+'.eps', format=eps)
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
