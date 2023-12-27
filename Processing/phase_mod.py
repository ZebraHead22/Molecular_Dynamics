import math
import numpy as np
import scipy as sp
from scipy import fftpack
from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt

t = np.arange(0, 0.5*10**-12, 10**-15)
fc = 100*(10**12)

s_1 = np.sin(2*math.pi*fc*t)
s_2 = np.sin(2*math.pi*fc*t+math.pi/12)
s_3 = np.sin(2*math.pi*fc*t+5*math.pi/6)
s_4 = np.sin(3*math.pi*fc*t+math.pi)
p_mod = s_1 + s_2 + s_3 + s_4

fft = sp.fftpack.fft(p_mod)
psd = np.abs(fft)
fftFreq = sp.fftpack.fftfreq(len(fft))
i = fftFreq > 0

# plt.plot(t, s_1)
# plt.plot(t, s_2)
# plt.plot(t, s_3)
# plt.plot(t, s_4)
# plt.plot(t, p_mod)
plt.scatter(fftFreq[i], psd[i])
plt.grid()
plt.show()
