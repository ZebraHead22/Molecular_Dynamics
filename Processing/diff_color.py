# Здесь строим кучу спектров из набора дат файлов в одной папке
import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

wv_1 = list()
wv_2 = list()
x_samples_1 = list()
x_samples_2 = list()
y_samples_1 = list()
y_samples_2 = list()

# --------------------------------------------------------------------------------------------
files_1 = os.listdir(os.getcwd())
for i in files_1:
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
        df.rename(columns={'0.0': 'Frequency',
                    '0.0.1': 'Amplitude'}, inplace=True)

        frequency = re.search(r'\d+', str(os.path.basename(filename)))
        frequency = frequency.group(0)
        closest_value_min = df.iloc[(
            df['Frequency']-float(int(frequency)-20)).abs().argsort()[:1]].index.tolist()
        closest_value_max = df.iloc[(
            df['Frequency']-float(int(frequency)+20)).abs().argsort()[:1]].index.tolist()
        max_amplitude = df.loc[closest_value_min[0]: closest_value_max[0], 'Amplitude'].max()
        max_amplitude_frequency = df.loc[df['Amplitude']
                                         == max_amplitude, 'Frequency']
        x_samples_1.append(max_amplitude_frequency)
        y_samples_1.append(max_amplitude)

x_samples_1 = [float(x) for x in x_samples_1]
for i in x_samples_1:
    wv_1.append((1/i)*10**4)
    # wv_1.append(i)

# --------------------------------------------------------------------------------------------
files_2 = os.listdir(os.getcwd()+"/Literature/")
for i in files_2:
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        df = pd.read_csv(os.getcwd()+'/Literature/'+i, delimiter=' ', index_col=None)
        df.rename(columns={'0.0': 'Frequency',
                    '0.0.1': 'Amplitude'}, inplace=True)

        frequency = re.search(r'\d+', str(os.path.basename(filename)))
        frequency = frequency.group(0)

        closest_value_min = df.iloc[(
            df['Frequency']-float(int(frequency)-20)).abs().argsort()[:1]].index.tolist()
        closest_value_max = df.iloc[(
            df['Frequency']-float(int(frequency)+20)).abs().argsort()[:1]].index.tolist()
        max_amplitude = df.loc[closest_value_min[0]: closest_value_max[0], 'Amplitude'].max()
        max_amplitude_frequency = df.loc[df['Amplitude']
                                         == max_amplitude, 'Frequency']
        x_samples_2.append(max_amplitude_frequency)
        y_samples_2.append(max_amplitude)

x_samples_2 = [float(x) for x in x_samples_2]
for i in x_samples_2:
    wv_2.append((1/i)*10**4)
    # wv_2.append(i)
# # --------------------------------------------------------------------------------------------
plt.gcf().clear()
plt.scatter(np.array(wv_1), np.array(y_samples_1), color = 'red')
plt.scatter(np.array(wv_2), np.array(y_samples_2), color  = 'darkblue')

plt.ylabel('Spectral Density (a.u.)')
plt.xlabel('Wavelenght ($\mu$m)')
plt.legend(['New data','Literature data'], loc=1)

plt.grid()
plt.show()
# plt.savefig("dep.png")
