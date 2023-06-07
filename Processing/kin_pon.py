# Import modules
import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Lists
time_kin_2 = list()
time_kin_4 = list()
time_kin_6 = list()
time_kin_8 = list()
time_pon_2 = list()
time_pon_4 = list()
time_pon_6 = list()
time_pon_8 = list()

kin_2 = list()
kin_4 = list()
kin_6 = list()
kin_8 = list()
pon_2 = list()
pon_4 = list()
pon_6 = list()
pon_8 = list()
# Main func
directory = os.getcwd()
for address, dirs, names in os.walk(directory):
    for name in names:
        filename, ext = os.path.splitext(name)
        if ext == '.dat':
            df = pd.read_csv(os.path.join(address, name),
                             delimiter=' ', index_col=None, header=None)
            df.rename(columns={0: 'TS', 1: 'Amplitude'}, inplace=True)
            df = df.drop_duplicates(keep='last')
            df.insert(1, "Time", (df['TS'] * 0.001))
            df.insert(3, "Energy", (df['Amplitude'] * 0.0434/5400))

            field_amplitude = re.search(r'\d', filename)
            field_amplitude = field_amplitude.group(0)
            energy_type = re.search(r'\w{3}$', filename)
            energy_type = energy_type.group(0)

            time = df['Time'].values.tolist()
            energy = df['Energy'].values.tolist()

            if energy_type == 'kin':
                if field_amplitude == '2':
                    kin_2 = energy
                    time_kin_2 = time
                elif field_amplitude == '4':
                    kin_4 = energy
                    time_kin_4 = time
                elif field_amplitude == '6':
                    kin_6 = energy
                    time_kin_6 = time
                elif field_amplitude == '8':
                    kin_8 = energy
                    time_kin_8 = time
            else:
                if field_amplitude == '2':
                    pon_2 = energy
                    time_pon_2 = time
                elif field_amplitude == '4':
                    pon_4 = energy
                    time_pon_4 = time
                elif field_amplitude == '6':
                    pon_6 = energy
                    time_pon_6 = time
                elif field_amplitude == '8':
                    pon_8 = energy
                    time_pon_8 = time


# Plotting and save figs
plt.gcf().clear()
fig, ax = plt.subplots(nrows=1, ncols=2)


def cm_to_inch(value):  # Define picture size, calc cm in inch
    return value/2.54


fig.set_figheight(cm_to_inch(10))
fig.set_figwidth(cm_to_inch(30))

ax[0].scatter(time_kin_2, kin_2, c='#57BBEB', s=2)
ax[0].scatter(time_kin_4, kin_4, c='#88EA3F', s=2)
ax[0].scatter(time_kin_6, kin_6, c='#DE28EB', s=2)
ax[0].scatter(time_kin_8, kin_8, c='#EB9834', s=2)

ax[1].scatter(time_pon_2, pon_2, c='#57BBEB', s=2)
ax[1].scatter(time_pon_4, pon_4, c='#88EA3F', s=2)
ax[1].scatter(time_pon_6, pon_6, c='#DE28EB', s=2)
ax[1].scatter(time_pon_8, pon_8, c='#EB9834', s=2)

ax[0].set_xlabel('Field amplitude (V/nm)')
ax[1].set_xlabel('Field amplitude (V/nm)')
ax[0].set_ylabel('Energy (eV)')
ax[1].set_ylabel('Energy (eV)')
ax[0].grid()
ax[1].grid()
ax[0].title.set_text('Kinetic')
ax[1].title.set_text('Potential')
plt.legend(['0.087 V/nm', '0.173 V/nm', '0.260 V/nm', '0.346 V/nm'])
plt.savefig(directory+'/'+'dependence.png')
