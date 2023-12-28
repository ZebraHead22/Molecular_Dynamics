import numpy as np
import scipy as sp
from scipy import fftpack
from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt

F = 100.e14          # No. of cycles per second, 
T = 10.e-15         # Time period, T = 10 fs
Fs = 1.e17        # No. of samples per second, 
Ts = 1./Fs        # Sampling interval, 
N = int(T/Ts)     # No. of samples for 2 ms, N = 100

t = np.linspace(0, T, N)

f = np.sin(2*np.pi*100.e14*t) #Функция для флуктуации
signal = np.sin(2*np.pi*F*t+f)
signal2 = 0.3*np.sin(2*np.pi*F*t+2*f)
s = signal*signal2

fft = sp.fftpack.fft(s)
psd = np.abs(fft)
fftFreq = sp.fftpack.fftfreq(len(fft))
i = fftFreq > 0
plt.plot(fftFreq[i], psd[i])
# plt.plot(t, signal2)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid()
plt.show()