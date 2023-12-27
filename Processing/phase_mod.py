import math
import numpy as np
import scipy as sp
from scipy import fftpack
from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt

t = np.arange(0, 10**-12, 10**-15)
fc = 100*(10**12)
fc1 = 100.5*(10**12)
fc2 = 100.1*(10**12)
fc3 = 100.9*(10**12)
s_1 = np.sin(2*math.pi*fc*t)
s_2 = 0.1*np.sin(2*math.pi*fc1*t)
s_3 = 0.6*np.sin(2*math.pi*fc2*t+5*math.pi/6)
s_4 = np.sin(2*math.pi*fc3*t+math.pi)
p_mod = s_1*s_2*s_3*s_4
# p_mod = list(map(, zip(s_1, s_2, s_3, s_4)))

fft = sp.fftpack.fft(p_mod)
psd = np.abs(fft)
fftFreq = sp.fftpack.fftfreq(len(fft))
i = fftFreq > 0

# plt.plot(t, s_1)
# plt.plot(t, s_2)
# plt.plot(t, s_3)
# plt.plot(t, s_4)
# plt.plot(t, p_mod)
plt.plot(fftFreq[i], psd[i])
plt.grid()
# plt.ylim([0, 0.01])
# plt.xlim([0.1995, 0.2005])
plt.show()
