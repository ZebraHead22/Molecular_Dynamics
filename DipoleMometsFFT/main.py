import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy import signal

filename = './testData/dipole_ff_1s.dat'
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
# plt.plot(x, y)
# plt.xlabel('time')
# plt.ylabel('dip')
# plt.grid()
# plt.show()

fs = 1 / timeframe  # частота дискретизаци, задается пользователем на основе снятых данных

f, SPM = sc.signal.welch(y, fs, 'cosine', nperseg=1024, average='median')  # построение СПМ с использованием окна
f1, SPM_w = sc.signal.welch(y, fs, nperseg=1024, average='median')  # СПМ без окон
f1 = f1 / (300 * 10 ** 8)
f = f / (300 * 10 ** 8)  # перевод частот в волновое число, для удобства
plt.semilogy(f, SPM)
plt.semilogy(f1, SPM_w)
plt.legend(['with window', 'without window'])
plt.xlim(0, 6000)  # в каком диапазоне частот строить график, модифицирую
plt.xlabel('Frequency ($cm^{-1}$)')
plt.ylabel('PSD (a.u)')
plt.grid()
plt.show()
