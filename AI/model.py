import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_spectrum(num_points=200, num_true_peaks=5, noise_level=0.1):
    spectrum = np.zeros(num_points)
    
    # Генерация истинных пиков
    true_peak_indices = np.random.choice(num_points, size=num_true_peaks, replace=False)
    for idx in true_peak_indices:
        spectrum[idx] = np.random.uniform(1.0, 2.0)  # Случайная высота пика
    
    # Добавление шума
    noise = np.random.normal(scale=noise_level, size=num_points)
    spectrum += noise
    
    # Добавление ложных пиков
    false_peak_indices = np.random.choice([i for i in range(num_points) if i not in true_peak_indices], 
                                          size=num_true_peaks * 2, replace=False)
    for idx in false_peak_indices:
        spectrum[idx] = np.random.uniform(0.1, 0.5)  # Меньшая случайная высота для ложного пика
    
    return spectrum, true_peak_indices

def plot_spectrum(spectrum, true_peak_indices, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(spectrum, label='Спектр')
    
    # Пометка истинных пиков
    for idx in true_peak_indices:
        plt.plot(idx, spectrum[idx], 'rx', markersize=10)
    
    # Пометка ложных пиков
    false_peak_indices = [i for i in range(len(spectrum)) if i not in true_peak_indices]
    for idx in false_peak_indices:
        if spectrum[idx] > 0.1:  # Пропуск точек, которые близки к нулю
            plt.plot(idx, spectrum[idx], 'bx', markersize=5)
    
    plt.xlabel('Частота')
    plt.ylabel('Интенсивность')
    plt.legend()
    plt.title('Спектр с пиками')
    plt.savefig(filename)
    plt.close()  # Закрываем фигуру, чтобы освободить память

def save_to_csv(spectrum, true_peak_indices, filename):
    data = []
    for i, value in enumerate(spectrum):
        if i in true_peak_indices:
            label = 1
        else:
            label = 0
        data.append((i, value, label))
    
    df = pd.DataFrame(data, columns=['Частота', 'Интенсивность', 'Пик (1 - Истинный, 0 - Ложный)'])
    df.to_csv(filename, index=False)

def main():
    num_spectra = 1000
    picture_dir = 'model/pictures'
    data_dir = 'model/data'

    # Создаем директории, если они не существуют
    os.makedirs(picture_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    for i in range(num_spectra):
        spectrum, true_peak_indices = generate_spectrum()
        
        # Сохранение спектра в CSV
        save_to_csv(spectrum, true_peak_indices, os.path.join(data_dir, f'spectrum_{i+1}.csv'))
        
        # Построение спектра и сохранение изображения
        plot_spectrum(spectrum, true_peak_indices, os.path.join(picture_dir, f'spectrum_{i+1}.png'))

if __name__ == "__main__":
    main()