import os
import re
import time
import threading
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
        num_cores = cpu_count()  # Use all available cores if not specified

    # Splitting into smaller chunks for better load balancing
    chunk_size = max(1000, N // (num_cores * 10))  # Minimum 1000 elements per chunk
    ranges = [(i, min(i + chunk_size, N)) for i in range(0, N, chunk_size)]

    with Pool(processes=num_cores) as pool:
        # Use map_async for dynamic task distribution
        results = pool.map_async(autocorrelation_chunk, [(dip_magnitude, start, end) for start, end in ranges]).get()

    # Summing up the autocorrelation results
    autocorr = np.zeros(len(dip_magnitude))
    for result in results:
        autocorr[:len(result)] += result

    return autocorr

def spinner():
    while True:
        for cursor in '|/-\\':
            yield cursor

def spinning_cursor(spinner_generator):
    while True:
        print(f'\r--Processing... {next(spinner_generator)}', end='', flush=True)
        time.sleep(0.1)

if __name__ == '__main__':
    directory = os.getcwd()
    for address, dirs, names in os.walk(directory):
        for name in names:
            filename, file_extension = os.path.splitext(name)
            if file_extension == ".dat":
                print(f"--File: {os.path.basename(filename)}")
                title = re.search(r'\d+\w+', os.path.basename(filename))
                title = title.group(0)
                df = pd.read_csv(
                        directory + "/" + name, delimiter=' ', index_col=None, header=[0])
                df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                    'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
                df.dropna(how='all', axis=1, inplace=True)
                dip_magnitude = np.array(df['|dip|'].to_list())
                dip_magnitude = dip_magnitude - np.mean(dip_magnitude)
                print(f"--Len of transent: {len(df['|dip|'].to_list())}")
                
                num_cores_to_use = 4  # Specify the number of cores to use
                total_cores = cpu_count()
                print(f"--Available CPU cores: {total_cores}")
                print(f"--Number of cores used for the task: {num_cores_to_use}")

                spinner_gen = spinner()
                spinner_thread = threading.Thread(target=spinning_cursor, args=(spinner_gen,))
                spinner_thread.start()

                try:
                    # Calculating the autocorrelation function
                    dip_magnitude_corr = calculate_autocorrelation(dip_magnitude, num_cores=num_cores_to_use)

                    # Applying the Hamming window
                    window = hann(len(dip_magnitude_corr))
                    dip_magnitude_windowed = dip_magnitude_corr * window

                    # Time step between points (in seconds)
                    time_step = 2e-15

                    # Fourier Transform
                    N = len(dip_magnitude_windowed)
                    yf = fft(dip_magnitude_windowed)
                    xf = fftfreq(N, time_step)[:N//2]

                    # Applying high-pass filter
                    cutoff_frequency = 1e12  # Cutoff frequency in Hz (e.g., 1 THz)
                    cutoff_index = np.where(xf < cutoff_frequency)[0][-1] + 1  # Index of the last frequency below cutoff
                    yf[:cutoff_index] = 0  # Zeroing out low-frequency components
                    
                    # Converting frequency from Hz to THz
                    xf_thz = xf * 1e-12

                    # Converting frequency from THz to cm^-1
                    xf_cm_inv = xf_thz / 0.03

                    # Selecting data up to 6000 cm^-1
                    mask = xf_cm_inv <= 6000
                    xf_cm_inv_filtered = xf_cm_inv[mask]
                    spectral_density_filtered = 2.0/N * np.abs(yf[:N//2])[mask]

                finally:
                    # Stop the spinner once processing is done
                    spinner_thread.do_run = False
                    spinner_thread.join()

                print("\nProcessing complete.")

                # Plotting the graph up to 6000 cm^-1
                plt.gcf().clear()
                plt.plot(xf_cm_inv_filtered, spectral_density_filtered, c='black')
                plt.xlim(0, 6000)
                plt.xlabel('Frequency ($cm^{-1}$)')
                plt.ylabel('Spectral Amplitude (a.u.)')
                plt.title(os.path.basename(filename))
                plt.grid()
                plt.savefig(filename + '_ac.png', dpi=600)
