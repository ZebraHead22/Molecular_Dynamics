import numpy as np, matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft
import os
import re
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy import signal


file = "/Users/max/Yandex.Disk.localized/NamdData/noField/glycine_5fs/dipole.dat"
df = pd.read_csv(file, sep = ' ')
df.dropna(how='all', axis=1, inplace=True)
df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': 'dip_abs'}, inplace=True)
df.insert(1, "Time", (df['frame']*10**(-15)))
f0 = rfft(df['dip_abs'].tolist())
f0 = np.abs(f0)
f1 = f0.copy()
f1[:3330] = 0
z = irfft(f1)

fftFreq = sc.fftpack.fftfreq(len(df['Time'].tolist()), 5*10**(-15))

i = fftFreq > 0

reverseCm = 1/((3*(10**10))/(fftFreq[i]))

f, (dip1, dip2) = plt.subplots(2, 1, sharey=False)

dip1.plot(np.array(np.array(fftFreq[i])), f0[i], label='dip1')
dip1.title.set_text('dip1')

dip2.plot(np.array(fftFreq[i]), z[i], label='dip2')
dip2.title.set_text('dip2')

plt.show()
