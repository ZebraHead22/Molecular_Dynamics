import os
import re
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy import signal

timeframe = input("Укажите время разделения фреймов (фс): ")
timeframe = float(float(timeframe) * (10 ** -15))
fieldtime = int(input("Укажите время действия поля (пс): "))

files = os.listdir(os.getcwd())
dirPath = os.getcwd()
for i in files:
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        df = pd.read_csv(dirPath+'/'+i, sep = ' ')
        df.dropna(how='all', axis=1, inplace=True)
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': 'dip_abs'}, inplace=True)
        df.insert(1, "Time", (df['frame'] * timeframe)*10**12)
        yList = np.array((((df['dip_x'])**2)+((df['dip_y'])**2))**(1/2)) #(x^2+y^2)^1/2
        # yList = df['dip_x'] + df['dip_y'] * 1j
        xList = np.array(df["Time"])
        plt.gcf().clear()

        # x = [ele.real for ele in yList]
        # y = [ele.imag for ele in yList]

        if os.path.basename(filename) == "dipole_00":
            plt.scatter(xList, yList, c = 'darkblue', s = 15)
            # plt.scatter(np.array(x), np.array(y))
            plt.ylabel('Dipole moment (D)')
            plt.xlabel('Time (ps)')
            plt.ylim([-20, 20])
            plt.grid()
            plt.savefig(filename+'XY.png')
        else:
            plt.scatter(xList, yList, c = 'darkblue', s = 15)
            # plt.scatter(np.array(x), np.array(y))
            plt.vlines(fieldtime, -50, 50, color = 'r')
            plt.ylabel('Dipole moment (D)')
            plt.xlabel('Time (ps)')
            plt.ylim([-20, 20])
            plt.grid()
            plt.savefig(filename+'XY.png')
