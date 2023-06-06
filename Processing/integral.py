# Import modules
import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Creating lists
gly_field_amplitudes = list()
ff_field_amplitudes = list()
trp_field_amplitudes = list()
gly_x_low = list()
ff_x_low = list()
trp_x_low = list()
gly_x_all = list()
ff_x_all = list()
trp_x_all = list()

# Main func
directory = os.getcwd()

for address, dirs, files in os.walk(directory):
    for name in files:
        filename, extension = os.path.splitext(name)
        if extension == '.dat':
            dataframe = pd.read_csv(os.path.join(address, name), delimiter=' ', index_col=None)
            dataframe.rename(columns={'0.0': 'Frequency', '0.0.1': 'Amplitude'}, inplace=True)

            high_energy = dataframe['Amplitude'].where(dataframe['Frequency'] < 5699 ).sum()
            low_energy = dataframe['Amplitude'].where(dataframe['Frequency'] < 220).sum()
            
            field_amplitude = re.search(r'\d+', name)
            field_amplitude = field_amplitude.group(0)

            folder = re.search(r'\w+$', address)
            folder = folder.group(0)
            
            if str(folder) == 'gly':
                gly_field_amplitudes.append(float(field_amplitude)/10)
                gly_x_all.append(float(high_energy))
                gly_x_low.append(float(low_energy))
            elif str(folder) == 'ff':
                ff_field_amplitudes.append(float(field_amplitude)/10)
                ff_x_all.append(float(high_energy))
                ff_x_low.append(float(low_energy))
            elif str(folder) == 'trp':
                trp_field_amplitudes.append(float(field_amplitude)/10)
                trp_x_all.append(float(high_energy))
                trp_x_low.append(float(low_energy))
           
gly_field_amplitudes = [i*(433.*(10**(-3))) for i in gly_field_amplitudes]
ff_field_amplitudes = [i*(433.*(10**(-3))) for i in ff_field_amplitudes]
trp_field_amplitudes = [i*(433.*(10**(-3))) for i in trp_field_amplitudes]

# Plotting and save figs
plt.gcf().clear()
fig, ax = plt.subplots(nrows=1, ncols=2)

def cm_to_inch(value):  # Define picture size, calc cm in inch
    return value/2.54

fig.set_figheight(cm_to_inch(10))
fig.set_figwidth(cm_to_inch(30))

ax[0].scatter(gly_field_amplitudes, gly_x_all, c='white', s=40, 
              linewidths=2, edgecolors='red')
ax[0].scatter(ff_field_amplitudes, ff_x_all, c='white', s=40,
              linewidths=2, edgecolors='darkblue')
ax[0].scatter(trp_field_amplitudes, trp_x_all, c='white', s=40,
              linewidths=2, edgecolors='darkgreen')

ax[1].scatter(gly_field_amplitudes, gly_x_low, c='white', s=40,
              linewidths=2, edgecolors='red')
ax[1].scatter(ff_field_amplitudes, ff_x_low, c='white', s=40,
              linewidths=2, edgecolors='darkblue')
ax[1].scatter(trp_field_amplitudes, trp_x_low, c='white', s=40,
              linewidths=2, edgecolors='darkgreen')


ax[0].set_xlabel('Field amplitude (V/nm)')
ax[1].set_xlabel('Field amplitude (V/nm)')
ax[0].set_ylabel('Energy (a.u.)')
ax[1].set_ylabel('Energy (a.u.)')
ax[0].grid()
ax[1].grid()
plt.legend(['Glycine', 'Diphenylalanine', 'Tryptophan'])
plt.savefig(directory+'/'+'dependence.png')

print(ff_x_all)
