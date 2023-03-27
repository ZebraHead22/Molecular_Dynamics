# UTF-8
#import modules
import os
import re
import scipy as sc
import numpy as np
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
from scipy.fftpack import rfft, irfft
import numpy as np, matplotlib.pyplot as plt
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
file = "/Users/max/Yandex.Disk.localized/NamdData/impuls/glycine/2nd_realizaiton/dipole_10.dat"
df = pd.read_csv(file, sep = ' ') # Чтение csv
df.dropna(how='all', axis=1, inplace=True) # Сброс пустых столбцов
df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': 'dip_abs'}, inplace=True) # Переименование столбцов
df.insert(1, "Time", (df['frame']*10**(-15))) # Добавка временного столбца

window = np.hanning(int(round(len(df['frame'])))) # Пишем окно
ySamp = (df['dip_abs'].tolist()) * window # Здесь диполи с окном

dipoles = sc.fftpack.fft(np.array(ySamp)) # Считаем амплитуды спектра
dipoles = np.abs(dipoles) # Берем только положительные
fftFreq = sc.fftpack.fftfreq(len(df['Time'].tolist()), 1*10**(-15)) # Считаем частоты
i = fftFreq > 0 # Фильтруем положительные частоты
reverseCm = 1/((3*(10**10))/(fftFreq[i])) # Преобразуем частоты из ТГц в см-1

dipoles[:2000] = 0 # Удаляем первые точки, где максимальная бесполезная амплитуда
df["SA"] = dipoles # Добавляем в датафрейм новый столбец с данными о спектральных амплитудах
timeList = df["Time"].tolist() # Формируем обычный список времен для выобрки по индексам

maxTimeIndex = int(timeList.index(df.loc[df['SA'].idxmax(), "Time"])) # Ищем индекс времени в списке, соттветствующий максимальной амплитуде колебаний

dipolesCopy = dipoles.copy() # Создаем копию массива
newDipoles = irfft(dipolesCopy) # Обратное преобразование Фурье

plt.gcf().clear()
# plt.plot(np.array(df["Time"].tolist()), np.array(newDipoles))
plt.plot(df["Time"].tolist(), df["dip_x"].tolist())
plt.plot(df["Time"].tolist(), newDipoles)

# plt.xlim(0, 6000)
plt.show()

