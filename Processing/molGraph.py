import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# name = os.getcwd()
# basename = str(os.path.basename(name))
# filename, file_extension = os.path.splitext(name)
# files = os.listdir(name)

file = '/Users/max/Yandex.Disk.localized/NAMD/pAla_12_10/spec.dat'

class Specdata:
    def __init__(self, path):
        self.path = str(path)
        self.nornname = os.path.splitext(os.path.basename(path))[0]
        self.df = pd.read_csv(self.path, delimiter=' ', index_col=None)
        self.df.rename(columns={'0.0': 'Frequency',
                '0.0.1': 'Amplitude'}, inplace=True)
        self.mean_amplitude = self.df['Amplitude'].mean()
        
    def plot_graph(self):
        plt.plot(self.df['Frequency'], self.df['Amplitude'])
        plt.ylabel('Spectral Density (a.u.)')
        plt.xlabel('Frequency ($cm^{-1}$)')
        plt.grid()
        plt.savefig(self.nornname+'.png')

data = Specdata(file)
# data.plot_graph()