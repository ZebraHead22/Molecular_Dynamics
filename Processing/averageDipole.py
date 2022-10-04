import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

frames = []
dip_x_1 = []
dip_y_1 = []
dip_z_1 = []
dip_abs_1 = []

df = pd.DataFrame()
files = os.listdir(os.getcwd())

for i in files:
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        df = pd.read_csv(i, sep = ' ')
        df.dropna(how='all', axis=1, inplace=True)
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
        dip_x_1.append(df['dip_x'].tolist())
        dip_y_1.append(df['dip_y'].tolist())
        dip_z_1.append(df['dip_z'].tolist())
        dip_abs_1.append(df['|dip|'].tolist())

dip_x = []
dip_y = []
dip_z = []
dip_abs = []

for i in dip_abs_1:
    for j in i:
        dip_abs.append(j)

for i in dip_x_1:
    for j in i:
        dip_x.append(j)

for i in dip_y_1:
    for j in i:
        dip_y.append(j)

for i in dip_z_1:
    for j in i:
        dip_z.append(j)

for i in range(len(dip_x)):
    frames.append(int(i)/200)

dipole = pd.DataFrame(list(zip(frames, dip_x, dip_y, dip_z, dip_abs)),
               columns =['Frames', 'dip_x', 'dip_y', 'dip_z', '|dip|'])

with pd.ExcelWriter(filename+'.xlsx') as writer:                 
    dipole.to_excel(writer, sheet_name='Dipole', index=None, index_label=None)


plt.gcf().clear()
fig, axes = plt.subplots(2, 2)

axes[0][0].scatter(np.array(frames), np.array(dip_x),
                   marker = 's',
                   c = 'fuchsia')
axes[0][0].set_title('dip_x')


axes[0][1].scatter(np.array(frames), np.array(dip_y),
                   marker = '*',
                   c = 'deeppink',
                   s = 700)
axes[0][1].set_title('dip_y')


axes[1][0].scatter(np.array(frames), np.array(dip_z),
                   marker = 'o',
                   c = 'lightcoral',
                   s = 100,
                   linewidths = 2,
                   edgecolors = 'darkred')
axes[1][0].set_title('dip_z')


axes[1][1].scatter(np.array(frames), np.array(dip_abs),
                   marker = 'o',
                   c = 'magenta',
                   s = 100,
                   edgecolors = 'black',
                   alpha = 0.6)
axes[1][1].set_title('|dip|')

# fig.set_figwidth(12)    #  ширина и
# fig.set_figheight(12)    #  высота "Figure"

plt.show()

# plt.plot(np.array(frames), np.array(dip_abs),c = 'deeppink', marker = 'o', edgecolors='black', s=100)

# plt.plot(np.array(frames), np.array(dip_abs), c = 'red', label = '|dip|')
# plt.plot(np.array(frames), np.array(dip_x), c = 'blue', label = 'dip_x')
# plt.plot(np.array(frames), np.array(dip_y), c = 'yellow', label = 'dip_y')
# plt.plot(np.array(frames), np.array(dip_z), c = 'green', label = 'dip_z')
# plt.legend()
# plt.errorbar(np.array(df['Time'].tolist()), np.array(df['average'].tolist()), np.array(df['SKO'].tolist()), fmt='o', c='deeppink',size=100, ecolor='black')
# plt.ylabel('Dipole Moment (a.u.)')
# plt.xlabel('Time (ps)')
# plt.grid()
