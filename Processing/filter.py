import numpy as np, matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft
import os
import re
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy import signal


file = "C:/Users/baranov_ma/YandexDisk/NamdData/noField/gly_1fs/dipole_gly.dat"
df = pd.read_csv(file, sep = ' ')
df.dropna(how='all', axis=1, inplace=True)
df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': 'dip_abs'}, inplace=True)
df.insert(1, "Time", (df['frame']*10**(-15)))

window = np.hanning(int(round(len(df['frame']))))
ySamp = (df['dip_abs'].tolist()) * window

dipoles = sc.fftpack.fft(np.array(ySamp))
dipoles = np.abs(dipoles)
fftFreq = sc.fftpack.fftfreq(len(df['Time'].tolist()), 1*10**(-15))
i = fftFreq > 0
reverseCm = 1/((3*(10**10))/(fftFreq[i]))


f1 = dipoles.copy()
f1[:9000] = 0
f1[11000:]=0
z = irfft(f1)


f, (dip1, dip2) = plt.subplots(2, 1, sharey=False)

dip1.plot(np.array(reverseCm), f1[i], label='dip1')
dip1.title.set_text('dip1')
dip1.axis(xmin=0,xmax=6000)

dip2.plot(np.array(df['Time'].tolist()), z, label='dip2')
dip2.title.set_text('dip2')
dip2.axis(xmin=0.1*10**(-10),xmax=0.9*10**(-10))
plt.show()
