import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy import signal
from scipy import fftpack

filename = './testData/dipole_gly_1s.dat'
df = pd.read_csv(filename, sep="\s+", skiprows=1, names=['frame', 'dip_x', 'dip_y', 'dip_z', '|dip|'])
# timeframe = input("Укажите время разделения фреймов: ")
# timeframe = float(timeframe)
timeframe = 1 * (10 ** -15) #sec
df.insert(1, "time", df['frame'] * timeframe)
axis = '|dip|'
y = np.array(df[axis])
avr = np.mean(y)
y = y - avr  # удаление постоянной составляющей
x = np.array(df['time'])
#-------------------------------------------------------------------------------------------------------------
window = np.hanning(int(round(len(x)))) # Взял Хэннинга просто так, можно другое попробовать...
y_res = y * window
energies_fft_win = sc.fftpack.fft(np.array(y_res)) # C окном
energies_fft = sc.fftpack.fft(np.array(y)) # Без окна
energies_psd_win = np.abs(energies_fft_win) # С окном
energies_psd = np.abs(energies_fft) # Без окна
fftFreq = sc.fftpack.fftfreq(len(energies_fft), timeframe) # Ищем частоты

reverseCm = 1/((3*(10**10))/(fftFreq)) # ТГц >>> см^(-1)

plt.plot(reverseCm, energies_psd_win, 'darkmagenta') # С окном
plt.plot(reverseCm, energies_psd,'darkcyan') # Без

plt.legend(['No correction', 'Hanning'])
plt.xlim(0, 6000)
plt.xlabel('Frequency ($cm^{-1}$)')
plt.ylabel('PSD (a.u)')
plt.grid()
plt.show()