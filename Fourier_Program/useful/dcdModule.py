from scipy import fftpack
import pandas as pd
import scipy as sp
import numpy as np
import re
from matplotlib import pyplot as plt
#Переделываем файл с правильными интервалами
out = []
file = './test_files/dipole.dat'
file = open(file, 'r+')
for line in file.readlines():
    newLine = ' '.join(line.split())+'\n'
    out.append(newLine)
file.close()
out[0] = 'frame dip_x dip_y dip_z abs\n'
newFile = open('./dipoles.csv', 'w')
for string in out:
    newFile.write(string)
newFile.close()
#Работаем с фреймами
df = pd.read_csv('dipoles.csv', sep=' ')
frames = np.array(df['frame'].tolist())
dipMoment = np.array(df['abs'].tolist())
#Gauss
window = np.kaiser(int(round(len(frames))), 0.01*(len(frames)))
newY = dipMoment*window
#Делаем спектр
energies_fft = sp.fftpack.fft(np.array(newY))
energies_psd = np.abs(energies_fft)
fftFreq = sp.fftpack.fftfreq(len(energies_fft), 1)
i = fftFreq > 0

plt.plot(fftFreq[i], 10 * np.log10(energies_psd[i]))
plt.show()

