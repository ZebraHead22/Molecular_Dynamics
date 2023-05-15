import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Requesting dcd-time and field time
timeframe = input("Укажите время разделения фреймов (фс): ")
timeframe = float(float(timeframe) * (10 ** -15))
fieldtime = int(input("Укажите время действия поля (пс): "))
# Define external electric field
one_field_period_list = [1, 1, 1, 1, 1,
                         -1, -1, -1, -1, -1]  # Frequency is 3333cm-1
field_list = one_field_period_list*int(fieldtime*100)
# Make zeros components
zero_copm_list = [0]*len(field_list)
# Make electric vector
electric_array = list()
a = list(zip(field_list, zero_copm_list, zero_copm_list))
for i in a:
    electric_array.append(np.array(i))
# Read csv
df = pd.read_csv(
    "/Users/max/Yandex.Disk.localized/Journals/Микроэлектроника/data/gly/dipole_20_1.dat", sep=' ', index_col=None)
df.dropna(how='all', axis=1, inplace=True)
df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                   'Unnamed: 6': 'dip_z', 'Unnamed: 8': 'dip_abs'}, inplace=True)
df.insert(1, "Time", (df['frame'] * timeframe)*10**12)
# Make and add zeros part electric vector
without_field_ts = int(len(df['Time'])-int(len(electric_array)))
zeros_part = list()
for i in range(without_field_ts):
    zeros_part.append(np.array([0, 0, 0]))
    i += 1
electric_array = electric_array + zeros_part
# Make dipole vector
dipole_vector = list()
for index, row in df.iterrows():
    dipole_vector.append(np.array([row['dip_x'], row['dip_y'], row['dip_z']]))
# Calculate polarization
polarization = list()
for i in range(len(dipole_vector)):
    a = np.dot(dipole_vector[i], electric_array[i])
    b = np.dot(electric_array[i], electric_array[i])
    polarization.append(float(a/b))
# Make graph
plt.gcf().clear()
plt.plot(np.array(df['Time']), np.array(polarization))
plt.grid()
plt.xlabel("Time (ps)")
plt.ylabel("Polarization")
plt.show()
