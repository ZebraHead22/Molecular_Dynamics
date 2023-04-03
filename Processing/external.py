import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series
from scipy import fftpack
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq

time = np.array(list(range(0, 80)))
magnitude = np.array([1, 1, 1, 1, -1, -1, -1, -1]*10)

if len(time) == len(magnitude):
    print("True")
    plt.scatter(time, magnitude)
    plt.ylabel('Amplitude (kcal/mol*A*e)')
    plt.xlabel('Time (ps)')
    plt.grid()

    plt.savefig('timeDependence_4.png')

    energies_fft = sp.fftpack.fft(magnitude)
    energies_psd = np.abs(energies_fft)
    fftFreq = sp.fftpack.fftfreq(len(energies_fft), 1/10**15)
    i = fftFreq > 0
    reverseCm = 1/((3*(10**10))/(fftFreq[i]))

    plt.gcf().clear()
    plt.plot(reverseCm, energies_psd[i])
    plt.ylabel('Spectral Density (a.u.)')
    plt.xlabel('Frequency ($cm^{-1}$)')
    plt.grid()
    # plt.show()
    data = pd.DataFrame(list(zip(reverseCm, energies_psd[i])), columns=[
                        'Frequency', 'Amplitude'])
    # print(data.loc[data['Amplitude'].idxmax()])
    for i in data['Amplitude'].nlargest(2):
        print(data['Frequency'].where(data['Amplitude'] == i).dropna())
else:
    print("False")
    print("len time="+str(len(time)))
    print("len magnitude="+str(len(magnitude)))
