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

<<<<<<< HEAD
        # Спектральный анализ
        window = hann(n)
        yf = rfft(autocorr * window)
        xf = rfftfreq(n, d=2e-15)
=======
    autocorr = np.zeros(len(dip_magnitude))
    for result in results:
        autocorr[:len(result)] += result

    return autocorr

# Генерация заголовка на основе имени файла
def create_title(filename):
    """
    Generate a title string based on the filename.

    Parameters:
        filename (str): The name of the file.

    Returns:
        str: A formatted title based on the filename.
    """
    match = re.search(r'(\w+)_(\d+)_([a-zA-Z]+)', filename)
    if match:
        prefix = match.group(1).upper()
        number = match.group(2)
        environment = match.group(3).lower()

        if 'water' in environment:
            return f"{prefix} WATER N={number}"
        elif 'vac' in environment or 'vacuum' in environment:
            return f"{prefix} VACUUM N={number}"
        elif 'linear' in environment:
            return f"{prefix} LINEAR N={number}"
        elif 'cyclic' in environment:
            return f"{prefix} CYCLIC N={number}"
    return filename

# Поиск основных пиков с учетом ширины на половине амплитуды
def find_main_peaks(xf_cm_inv_filtered, spectral_density_filtered):
    ranges = [(i, i + 500) for i in range(0, 4000, 500)]
    peak_frequencies = []
    peak_amplitudes = []
    peak_widths_half_max = []

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

                # Расчет ширины на половине высоты
                results_half_max = peak_widths(sub_spectral_density, [peak], rel_height=0.5)
                width = results_half_max[0][0]
                peak_widths_half_max.append(width * (sub_xf[1] - sub_xf[0]))

    return peak_frequencies, peak_amplitudes, peak_widths_half_max

# Аннотация и сохранение данных пиков в файл
def annotate_and_save_peaks(output_file, prefix, filename, peak_frequencies, peak_amplitudes, peak_widths_half_max):
    """
    Annotate peak data and save it to a file.

    Parameters:
        output_file (file object): The file to write the peak data to.
        prefix (str): A prefix for the annotation.
        filename (str): The name of the file being processed.
        peak_frequencies (list): List of peak frequencies.
        peak_amplitudes (list): List of peak amplitudes.
        peak_widths_half_max (list): List of peak widths at half maximum.

    Returns:
        None
    """
    for freq, amp, width in zip(peak_frequencies, peak_amplitudes, peak_widths_half_max):
        try:
            output_file.write(f"{prefix}_{filename} -- {freq:.2f} -- {amp:.2f} -- {width:.2f}\n")
        except (ValueError, TypeError) as e:
            print(f"Error writing data for {filename}: {e}")

# Главная функция для обработки каждого файла
def process_file(name, output_file, num_cores):
    filename, file_extension = os.path.splitext(name)
    if file_extension == ".dat":
        print(f"-- File {os.path.basename(filename)}")
        title = create_title(filename)
        print(f"-- Generated title: {title}")

        df = pd.read_csv(name, delimiter=' ', index_col=None, header=[0])
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                           'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)

        df.insert(1, 'Time', df['frame']*1/1000)

        dip_magnitude = np.array(df['|dip|'].to_list())
        dip_magnitude -= np.mean(dip_magnitude)


        plt.figure()
        plt.plot(df['Time'].to_list(), dip_magnitude, color='black', linewidth=0.5)
        plt.title(title + ' Raw Data')
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (a.u.)")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{filename}_raw.png", dpi=300)


        length = dip_magnitude.size
        print(f"-- Len of transient {length} points or {length * 1 / 1000000} ns")

        print(f"-- Using {num_cores} cores")

        # === Spectrum with Autocorrelation ===
        dip_magnitude_corr = calculate_autocorrelation(dip_magnitude, num_cores=num_cores)
>>>>>>> 9ed3d001e4ff0b817d570de9c490344f549bd844
        
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

<<<<<<< HEAD
        return True
=======
        time_step = 1e-15
        N = len(dip_magnitude_windowed) #_windowed if Hann
        yf = fft(dip_magnitude_windowed) # too
        xf = fftfreq(N, time_step)[:N//2]
>>>>>>> 9ed3d001e4ff0b817d570de9c490344f549bd844

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