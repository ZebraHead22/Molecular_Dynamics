import math
import numpy as np
import scipy as sp
from scipy import fftpack
from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt

#Constants
A1 = 1 #Множитель амплитуды несущего сигнала
A2 = 0.3 #Множитель моделирующего
N = 3  #Коэффициэнт модуляции
omega0 = 100*(10**12) #частота несущего
omega1 = 10**19 #частота флуктуаций

#Functions
t = np.arange(0, 10**-9, 10**-15) #Time
f = np.sin(2*math.pi*omega1*t) #Функция для флуктуации
s1 = A1*np.sin(2*math.pi*omega0*t+f) #Несущий сигнал
s2 = A2*np.sin(N*2*math.pi*omega0*t+N*f) #Модулирующий сигнал
ss = s1*s2 #Сумма

#FFT
fft = sp.fftpack.fft(ss)
psd = np.abs(fft)
fftFreq = sp.fftpack.fftfreq(len(fft))
i = fftFreq > 0
#Plotting
# plt.plot(t, s1)
# plt.plot(t, s2)
# plt.plot(t, ss)
plt.plot(fftFreq[i], psd[i])
plt.grid()
# plt.ylim([0, 0.01])
# plt.xlim([0.1995, 0.2005])
plt.xlabel('Frequency')
plt.ylabel('Amplitude (a.u.)')
plt.show()
