import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series
from scipy import fftpack
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq
'''
Тестим реализации различных внешних ПЛОСКИХ полей
'''
impuls_time = list()
impuls_list = list()
amplitudes =list()
zeros = [0]*1000  # Start and ending RI
factor = 100  # Start since 1ps
# One impuls period 10fs (0.01ps)
one_impuls_list = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]

all = np.array(zeros + impuls_list + zeros)
time = np.array(list(range(0, len(all))))

for factor in range(1, 20, 1):
    
    impuls_list = one_impuls_list * factor
    all = np.array(zeros + impuls_list + zeros)
    time = np.array(list(range(0, len(all))))

    impuls_time.append(len(impuls_list)/1000)

    energies_fft = sp.fftpack.fft(all)
    energies_psd = np.abs(energies_fft)
    fftFreq = sp.fftpack.fftfreq(len(energies_fft), 1/10**15)
    i = fftFreq > 0
    reverseCm = 1/((3*(10**10))/(fftFreq[i]))

    data = pd.DataFrame(list(zip(reverseCm, energies_psd[i])), columns=[
                        'Frequency', 'Amplitude'])
    # print(data.loc[data['Amplitude'].idxmax(), 'Frequency'])
    
    amplitudes.append(data.loc[data['Amplitude'].idxmax(), 'Frequency'])
    
plt.gcf().clear()
plt.plot(np.array(impuls_time), np.array(amplitudes), c = 'darkblue')
# plt.ylabel('Spectral Density (a.u.)')
plt.ylabel('Frequency ($cm^{-1}$)')
plt.xlabel('Radio pulse duration (ps)')
plt.grid()
plt.show()
