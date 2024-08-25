import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from multiprocessing import Pool, cpu_count
from scipy.signal.windows import hamming, hann, blackman, bartlett

def autocorrelation_chunk(args):
    dip_magnitude, start, end = args
    chunk = dip_magnitude[start:end]
    autocorr = np.correlate(chunk, chunk, mode='full')
    return autocorr[len(chunk)-1:]

def calculate_autocorrelation(dip_magnitude, num_cores=None):
    N = len(dip_magnitude)
    if num_cores is None:
        num_cores = cpu_count()  # Если не указано, используем все доступные ядра

    # Разбиение на более мелкие куски для лучшей балансировки
    chunk_size = max(1000, N // (num_cores * 10))  # Минимум 1000 элементов на кусок
    ranges = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

    with Pool(processes=num_cores) as pool:
        # Используем map_async для динамического распределения задач
        results = pool.map_async(autocorrelation_chunk, [(dip_magnitude, start, end) for start, end in ranges]).get()

    # Суммирование результатов автокорреляции
    autocorr = np.zeros(len(dip_magnitude))
    for result in results:
        autocorr[:len(result)] += result

    return autocorr

if __name__ == '__main__':
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
                
                num_cores_to_use = 4  # Задайте нужное количество ядер
                total_cores = cpu_count()
                print(f"Доступное количество ядер процессора: {total_cores}")
                print(f"Количество ядер, используемых для задачи: {num_cores_to_use}")

                # Рассчет автокорреляционной функции
                dip_magnitude_corr = calculate_autocorrelation(dip_magnitude, num_cores=num_cores_to_use)
                dip_magnitude_corr = dip_magnitude_corr[len(dip_magnitude_corr)//2:]  # Взять только правую половину
                # plt.gcf().clear()
                # plt.plot(df['frame'], dip_magnitude_corr, c='black')
                # plt.show()

                # Применение окна Хэмминга
                window = hann(len(dip_magnitude_corr))
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
                plt.title(os.path.basename(filename))
                plt.grid()
                # plt.show()
                plt.savefig(filename + '_ac.png', dpi = 600)
