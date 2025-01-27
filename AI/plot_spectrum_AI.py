import os
import logging
import numpy as np
import pandas as pd
import gc
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import find_peaks, peak_widths
from scipy.signal.windows import hann
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Конфигурация
INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
JOBS = 16
MIN_PEAK_HEIGHT = 0.1
CUTOFF_FREQ = 3e12  # В Гц (3 ТГц)

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'processing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_file(file_path):
    """Обработка одного файла"""
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_prefix = os.path.join(OUTPUT_DIR, base_name)
        logger.info(f"Processing: {file_path}")

        # Чтение данных
        df = pd.read_csv(
            file_path,
            sep=' ',
            usecols=['#', 'Unnamed: 8'],
            dtype={'#': 'int32', 'Unnamed: 8': 'float32'}
        )

        # Подготовка данных
        time = df['#'].values * 2e-3
        signal = df['Unnamed: 8'].values.astype('float32')
        signal -= signal.mean()

        # Автокорреляция
        n = len(signal)
        fft_sig = rfft(signal, n=2*n)
        autocorr = irfft(fft_sig * np.conj(fft_sig), n=2*n)[:n].real
        autocorr /= np.max(autocorr)
        del fft_sig
        gc.collect()

        # Спектральный анализ
        window = hann(n)
        yf = rfft(autocorr * window)
        xf = rfftfreq(n, d=2e-15)
        
        # Фильтрация
        cutoff_idx = np.searchsorted(xf, CUTOFF_FREQ)
        yf[:cutoff_idx] = 0
        
        # Конвертация в cm⁻¹
        xf_cm = (xf * 1e-12) / 0.03
        mask = xf_cm <= 4000
        xf_filtered = xf_cm[mask]
        spectrum = 2.0 / n * np.abs(yf[:len(mask)][mask])
        spectrum = np.array([i * 1000 for i in spectrum])

        # Поиск пиков
        peaks, _ = find_peaks(spectrum, height=MIN_PEAK_HEIGHT, width=2)
        widths = peak_widths(spectrum, peaks, rel_height=0.5)[0] * (xf_filtered[1] - xf_filtered[0]) if len(peaks) > 0 else []

        # Сохранение данных
        np.savez_compressed(
            f"{output_prefix}_results.npz",
            time=time,
            autocorrelation=autocorr,
            frequencies=xf_filtered,
            spectrum=spectrum,
            peaks=peaks,
            peak_widths=widths
        )

        # Построение графиков
        plt.figure(figsize=(12,6))
        plt.plot(time, signal, 'k-', lw=0.5)  # Черный цвет
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (a. u.)")
        plt.grid(True)
        plt.savefig(f"{output_prefix}_signal.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12,6))
        plt.plot(xf_filtered, spectrum, 'k-', lw=0.5)  # Черный цвет
        plt.xlabel("Frequency (cm⁻¹)")
        plt.ylabel("Spectral ACF EDM Amplitude (a. u.)")
        plt.grid(True)
        plt.savefig(f"{output_prefix}_spectrum.png", dpi=300, bbox_inches='tight')
        plt.close()

        return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    # Поиск .dat файлов
    dat_files = [
        os.path.join(INPUT_DIR, f) 
        for f in os.listdir(INPUT_DIR) 
        if f.endswith('.dat')
    ]

    if not dat_files:
        logger.error("No .dat files found in current directory!")
        exit(1)

    logger.info(f"Found {len(dat_files)} .dat files")
    logger.info("Processing parameters:")
    logger.info(f"* Number of processes: {JOBS}")
    logger.info(f"* Cutoff frequency: {CUTOFF_FREQ/1e12:.2f} THz")  # Конвертация в ТГц

    # Обработка файлов
    with Pool(min(JOBS, len(dat_files))) as pool:
        results = pool.map(process_file, dat_files)
        success = sum(results)
        logger.info(f"Successfully processed: {success}/{len(dat_files)} files")