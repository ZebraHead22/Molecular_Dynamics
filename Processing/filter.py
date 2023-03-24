import numpy as np, matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft
import os
import re
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy import signal


file = "C:/Users/baranov_ma/YandexDisk/NamdData/noField/glycine_5fs/dipole.dat"
df = pd.read_csv(file, sep = ' ')
df.dropna(how='all', axis=1, inplace=True)
df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': 'dip_abs'}, inplace=True)
df.insert(1, "Time", (df['frame'] *10**(-3)))


# n = 2 ** 8
# x = np.linspace(0, 2 * np.pi, n)
# y = np.sin(x) + np.cos(x * 30)
# f0 = rfft(y)
# f1 = f0.copy()
# f1[:n // 8] = 0
# z = irfft(f1)
# f, (spectra, signal) = plt.subplots(2, 1, sharey=False)
# spectra.plot(x, f0, label='f0')
# spectra.plot(x, f1, label='f1')
# spectra.legend()
# spectra.title.set_text('spectra')
# signal.plot(x, y, label='y')
# signal.plot(x, z, label='z')
# signal.legend()
# signal.title.set_text('signal')
# plt.legend()
# plt.show()