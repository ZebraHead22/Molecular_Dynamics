import os
import re
import numpy as np
import scipy as sc
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib as mpl

for address, dirs, names in os.walk(os.getcwd()):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            a = re.search(r'\d+', name)
            a = a.group(0)
            print(name)
            df = pd.read_csv(os.getcwd() + '/' + name, sep=' ')
            df.dropna(how='all', axis=1, inplace=True)
            df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                    'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
            df.insert(1, "Time", (df['frame'] / 1000))
            plt.gcf().clear()
            plt.plot(np.array(df["Time"].tolist()), np.array(df["dip_x"].tolist()), color='#DB07B8', linewidth=2)
            plt.plot(np.array(df["Time"].tolist()), np.array(df["dip_y"].tolist()), color='#0777DB', linewidth=2)
            plt.plot(np.array(df["Time"].tolist()), np.array(df["dip_z"].tolist()), color='#DB8A07', linewidth=2)
            plt.plot(np.array(df["Time"].tolist()), np.array(df["|dip|"].tolist()), color='#38DB07', linewidth=2)
            plt.legend(["dip_x", "dip_y", "dip_z", "|dip|"])
            plt.ylabel('Spectral Density (a.u.)')
            plt.xlabel('Time (ps)')
            plt.grid()
            plt.savefig(os.getcwd() + "/" + 'dipoleplot' + str(a) + ".png")
            print("Ready" + " " + name)


