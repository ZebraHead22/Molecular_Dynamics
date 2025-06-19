import os
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

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
        elif 'gbis' in environment:
            return f"{prefix} WATER (GBIS) N={number}"
        elif 'linear' in environment:
            return f"{prefix} LINEAR N={number}"
        elif 'cyclic' in environment:
            return f"{prefix} CYCLIC N={number}"
    return filename

def find_main_peaks(xf_cm_inv_filtered, spectral_density_filtered):
    ranges = [(i, i + 500) for i in range(0, 4000, 500)]
    # ranges = [(i, i + 500) for i in range(0, 4000, 500)]
    peak_frequencies = []
    peak_amplitudes = []

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

    return peak_frequencies, peak_amplitudes

def annotate_peaks(ax, peak_frequencies, peak_amplitudes, xlim_max=6000):
    peak_frequencies = np.array(peak_frequencies)
    peak_amplitudes = np.array(peak_amplitudes)
    max_peak_amplitude = np.max(peak_amplitudes)
    ylim_lower, ylim_upper = ax.get_ylim()

    def add_annotation(ax, freq, amp, label, offset_y, offset_x=0, draw_arrow=False):
        text_y = amp + offset_y
        if text_y > ylim_upper:
            text_y = ylim_upper - (ylim_upper * 0.05)
        if freq + offset_x > xlim_max:
            offset_x = xlim_max - freq

        ax.text(freq + offset_x, text_y, label, fontsize=12, rotation=0, ha='center')
        if draw_arrow:
            ax.annotate('', xy=(freq, amp), xytext=(freq + offset_x, text_y),
                        arrowprops=dict(facecolor='none', edgecolor='red', linestyle='dashed', 
                                        lw=0.5, shrink=0.05))

    legend_entries = []
    peak_counter = 1

    for freq, amp in zip(peak_frequencies, peak_amplitudes):
        add_annotation(ax, freq, amp, f'{peak_counter}', offset_y=max_peak_amplitude * 0.06)
        legend_entries.append(f'{peak_counter}: {freq:.2f} см⁻¹')
        peak_counter += 1

    legend_text = '\n'.join(legend_entries)
    ax.text(0.95, 0.95, legend_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.6))

    return ax

def process_file(filename, output_file):
    try:
        df = pd.read_csv(filename, delimiter=' ', index_col=None)
        df.rename(columns={'0.0': 'Frequency', '0.0.1': 'Amplitude'}, inplace=True)
        df.insert(2, 'Amp×104', df['Amplitude'] * (10**4))

        dfFreq = np.array(df['Frequency'].tolist())
        dfAmp = np.array(df['Amp×104'].tolist())

        peak_frequencies, peak_amplitudes = find_main_peaks(dfFreq, dfAmp)

        for freq, amp in zip(peak_frequencies, peak_amplitudes):
            output_file.write(f"{os.path.basename(filename)} -- {freq:.2f} -- {amp:.2f}\n")

        return dfFreq, dfAmp, peak_frequencies, peak_amplitudes
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, None, None, None

def make_spectres():
    output_file = open("peak_data.txt", "w")
    files = os.listdir(os.getcwd())
    
    for i in files:
        filename, file_extension = os.path.splitext(i)
        if file_extension == ".dat":
            title = create_title(filename)
            dfFreq, dfAmp, peak_frequencies, peak_amplitudes = process_file(i, output_file)

            if dfFreq is None:
                continue

            plt.gcf().clear()
            fig, ax = plt.subplots()
            ax.plot(dfFreq, dfAmp, linewidth=1, c='black')
            ax.grid()
            # ax.set_xlim([800, 1200])
            ax.set_title(title)
            # ax.set_xlabel('Frequency, $cm^{-1}$')
            ax.set_xlabel('Частота, $cм^{-1}$')
            # ax.set_ylabel('Spectral EDM Amplitude (a.u.)')
            ax.set_ylabel('Спектральная ЭДМ Амплитуда, отн. ед.')

            annotate_peaks(ax, peak_frequencies, peak_amplitudes)
            # plt.savefig(filename + '.png', dpi=600)
            # plt.savefig(filename + '.eps', format='eps')
            plt.savefig(filename+'.png', dpi=300)

    output_file.close()



            
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
