import os
import re
import numpy as np
import scipy as sp
import pandas as pd
from scipy import signal
from scipy import fftpack
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.ticker import LinearLocator, FormatStrFormatter

amino_acids = list()
energy_type = ['KINETIC', 'POTENTIAL']
field_amplitudes = [0.0435,	0.087,	0.1305,	0.174,	0.2175,
                    0.261,	0.3045,	0.348,	0.3915,	0.435,	0.4785,	0.522]

data = pd.DataFrame()
directory = os.getcwd()
for address, dirs, names in os.walk(directory):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            amino_acid = re.search(r'^\w{,2}[^\_]', filename)
            amino_acid = amino_acid.group(0)
            amino_acids.append(amino_acid)
amino_acids = list(set(amino_acids))

for i in amino_acids:
    directory = os.getcwd()
    directory = directory+"/"+i
    for j in energy_type:
        data = pd.DataFrame()
        for address, dirs, names in os.walk(directory):
            for name in names:
                filename, file_extension = os.path.splitext(name)
                if file_extension == ".dat":
                    one_data = pd.read_csv(os.path.join(
                        address, name), delimiter=' ', index_col=None, header=[0])
                    data["TIME"] = (one_data["TS"]) * 0.001
                    data[(str(filename)+"_"+j)] = (one_data[j]) * 0.0434/5400
        # print(data.head())
        # data.to_excel(directory+"_"+j+".xlsx")
        """Вычисление спектров энергий"""
        spectre_data = pd.DataFrame()
        for energy_column in data.columns.values[1:]:
            energies_psd = np.abs(sp.fftpack.fft(data[energy_column].tolist()))
            n = len(energies_psd)
            window = sp.signal.windows.gaussian(n, std=5000, sym=False)
            module_w = energies_psd
            b, a = sp.signal.butter(2, 0.05)
            filtered = sp.signal.filtfilt(b, a, module_w)
            fftFreq = sp.fftpack.fftfreq(len(data[energy_column].tolist()), 1/10**15)
            i = fftFreq > 0
            reverseCm = 1/((3*(10**10))/(fftFreq[i]))
            spectre_data["FREQUENCY"] = reverseCm
            spectre_data[(str(energy_column))] = filtered[i]
        print(spectre_data.head())
        spectre_data.to_excel(directory+"_"+j+"_spectre.xlsx")