import os
import re
import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy import fftpack
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq

for path, dirs, files in os.walk(os.getcwd()):
    for file in files:
        file_name, file_extension = os.path.splitext(path+'/'+file)
        if file_extension == '.csv':
            log_data = pd.read_csv(path+'/'+file, delimiter=',', index_col=None)
            log_data.insert(1, 'TIME', log_data['TS']/1000)
            log_data['SUM']=log_data.iloc[:,2:7].sum(axis=1)
            print(log_data.head())

energies_fft = sp.fftpack.fft(np.array(log_data['SUM'].tolist()))
energies_psd = np.abs(energies_fft)
fftFreq = sp.fftpack.fftfreq(len(energies_psd), 5/10**15)
i = fftFreq > 0
reverseCm = 1/((3*(10**10))/(fftFreq[i]))

plt.gcf().clear()
# plt.plot(np.array(log_data['TIME'].tolist()), np.array(log_data['SUM'].tolist()))
plt.plot(reverseCm, energies_psd[i])
plt.ylabel('Spectral Density (a.u.)')
plt.xlabel('Frequency ($cm^{-1}$)')
plt.grid()
plt.savefig(path+'/'+'mechanic_spectre.png', dpi=800)