import os
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
            df = pd.read_csv(
                    directory+ "/" +name, delimiter=' ', index_col=None, header=[0])
            df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                  'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
            df.dropna(how='all', axis=1, inplace=True)
            dip_magnitude = np.array(df['|dip|'].to_list())
            dip_magnitude = dip_magnitude - np.mean(dip_magnitude)

            # Применение окна Хэмминга
            window = hamming(len(dip_magnitude))
            dip_magnitude_windowed = dip_magnitude * window

            # Временное расстояние между точками (в секундах)
            time_step = 2e-15

            # Преобразование Фурье
            N = len(dip_magnitude_windowed)
            yf = fft(dip_magnitude_windowed)
            xf = fftfreq(N, time_step)[:N//2]

            yf1 = np.abs(yf[:N//2])
            yf1[:2000] = 0
            yf1 = [x * 10000 for x in yf1]

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
            plt.plot(xf_cm_inv, 2.0/N * np.abs(yf1), c='black')
            # plt.xlim(0, 6000)
            plt.xlabel('Frequency ($cm^{-1}$)')
            plt.ylabel('Spectral Amplitude (a.u. ×$10^{4}$)')
            # plt.title(name)
            plt.grid()
            # plt.show()
            plt.xlim(0, 1000)
            plt.savefig(filename + '_1000.png')
            plt.xlim(1000, 2000)
            plt.savefig(filename + '_2000.png')
            plt.xlim(2000, 3000)
            plt.savefig(filename + '_3000.png')
            plt.xlim(3000, 4000)
            plt.savefig(filename + '_4000.png')