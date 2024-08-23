import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hamming, hann, blackman, bartlett

directory = os.getcwd()
for address, dirs, names in os.walk(directory):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            title = re.search(r'\d+\w+', os.path.basename(filename))
            title = title.group(0)
            df = pd.read_csv(
                    directory+ "/" +name, delimiter=' ', index_col=None, header=[0])
            df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                  'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
            df.dropna(how='all', axis=1, inplace=True)
            dip_magnitude = np.array(df['|dip|'].to_list())
            dip_magnitude = dip_magnitude - np.mean(dip_magnitude)
            
            # Рассчет автокорреляционной функции
            dip_magnitude_corr = np.correlate(dip_magnitude, dip_magnitude, mode='full')
            dip_magnitude_corr = dip_magnitude_corr[len(dip_magnitude_corr)//2:]  # Взять только правую половину
            # plt.gcf().clear()
            # plt.plot(df['frame'], dip_magnitude_corr, c='black')
            # plt.show()

            # Применение окна Хэмминга
            window = hann(len(dip_magnitude))
            dip_magnitude_windowed = dip_magnitude_corr * window
            # plt.gcf().clear()
            # plt.plot(df['frame'], dip_magnitude_windowed, c='black')
            # plt.show()

            # Временное расстояние между точками (в секундах)
            time_step = 2e-15

            # Преобразование Фурье
            N = len(dip_magnitude_windowed)
            yf = fft(dip_magnitude_windowed)
            xf = fftfreq(N, time_step)[:N//2]

            # Применение фильтра высоких частот
            cutoff_frequency = 1e12  # Пороговая частота в Гц (например, 1 ТГц)
            cutoff_index = np.where(xf < cutoff_frequency)[0][-1] + 1  # Индекс последней частоты ниже порога
            yf[:cutoff_index] = 0  # Зануление низкочастотных компонентов
            
            # Конвертация частоты из Гц в ТГц
            xf_thz = xf * 1e-12

            # Конвертация частоты из ТГц в см^-1
            xf_cm_inv = xf_thz / 0.03

            # Выбор данных до 6000 см^-1
            mask = xf_cm_inv <= 6000
            xf_cm_inv_filtered = xf_cm_inv[mask]
            spectral_density_filtered = 2.0/N * np.abs(yf[:N//2])[mask]

            # Сохранение данных в файл
            # output_data = np.column_stack((xf_cm_inv_filtered, spectral_density_filtered))
            # output_file_path = filename + '_spectre.dat'
            # np.savetxt(output_file_path, output_data, fmt='%.6e', delimiter=' ', header='Frequency Amplitude', comments='')

            # Построение графика до 6000 см^-1
            plt.gcf().clear()
            plt.plot(xf_cm_inv_filtered, spectral_density_filtered, c='black')
            plt.xlim(0, 6000)
            plt.xlabel('Frequency ($cm^{-1}$)')
            # plt.ylabel('Spectral Amplitude (a.u. ×$10^{4}$)')
            plt.ylabel('Spectral Amplitude (a.u.)')
            plt.title(title)
            plt.grid()
            # plt.show()
            plt.savefig(filename + '.png', dpi = 600)
            
            # plt.xlim(0, 1000)
            # plt.savefig(filename + '_1000.png')
            # plt.xlim(1000, 2000)
            # plt.savefig(filename + '_2000.png')
            # plt.xlim(2000, 3000)
            # plt.savefig(filename + '_3000.png')
            # plt.xlim(3000, 4000)
            # plt.savefig(filename + '_4000.png')