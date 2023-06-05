import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

x_samples = list()
y_samples = list()
y2_samples = list()
directory = os.getcwd()

for address, dirs, files in os.walk(directory):
    for name in files:
        filename, extension = os.path.splitext(name)
        if extension == '.dat':
            dataframe = pd.read_csv(os.path.join(
                address, name), delimiter=' ', index_col=None)
            dataframe.rename(columns={'0.0': 'Frequency',
                                      '0.0.1': 'Amplitude'}, inplace=True)

            field_amplitude = re.search(r'\d+', name)
            field_amplitude = field_amplitude.group(0)
            x_samples.append(float(field_amplitude)/10)

            high_energy = dataframe['Amplitude'].where(
                dataframe['Frequency'] < 5500).sum()
            low_energy = dataframe['Amplitude'].where(
                dataframe['Frequency'] < 300).sum()
            y_samples.append(float(high_energy))
            y2_samples.append(float(low_energy))

x_samples = [i*(433.*(10**(-3))) for i in x_samples]
# y_samples = [i*(10**(-6)) for i in x_samples]

plt.gcf().clear()

fig, ax = plt.subplots(nrows=1, ncols=2)

def cm_to_inch(value):  # Define picture size, calc cm in inch
    return value/2.54
fig.set_figheight(cm_to_inch(10))  
fig.set_figwidth(cm_to_inch(26))  

ax[0].scatter(x_samples, y_samples, c='white', s=40,
            linewidths=2, edgecolors='red')
ax[1].scatter(x_samples, y2_samples, c='white', s=40,
            linewidths=2, edgecolors='darkblue')


ax[0].set_xlabel('Field amplitude (V/nm)')
ax[1].set_xlabel('Field amplitude (V/nm)')
ax[0].set_ylabel('Intergal energy')
ax[1].set_ylabel('Intergal energy')
ax[0].grid()
ax[1].grid()
plt.savefig(directory+'/'+'dep.png')
