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
legend = list()
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
    dat_files = list()
    for file in os.listdir(directory):
        dat_files.append(file)
    dat_files.sort()

    for j in energy_type:
        data = pd.DataFrame()
        for file in dat_files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == ".dat":
                one_data = pd.read_csv(directory+"/"+file, delimiter=' ', index_col=None, header=[0])
                data["TIME"] = (one_data["TS"]) * 0.001
                data[(str(filename)+"_"+j)] = (one_data[j]) * 0.0434/5400
        print(data.head())
        data.to_excel(directory+"_"+j+".xlsx")

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

        last_moment_energies = list()
        legend.append(os.path.basename(directory)+" "+j.lower())
        for energy_column in data.columns.values[1:]:
            last_moment_energies.append(float(data.iloc[-1, data.columns.get_loc(energy_column)]))
        plt.scatter(field_amplitudes, last_moment_energies, s=20)      
plt.grid()
plt.xlabel('Field amplitude (V/nm)')
plt.ylabel('Energy (eV)')
plt.title('Energy in 500ps time step \n(field amplitude)')
plt.legend(legend)
plt.savefig(directory+"/dependence.png")
       
       
        # plt.gcf().clear()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # # for y_column in data.drop('TIME', axis=1):
        # for z in field_amplitudes:
        #     x = data["TIME"]
        #     for y_column in data.iloc[:, 1:12]:
        #         y = np.array(data[y_column].to_list())

        #     def plot2din3d(x,y,z):
        #         ax.plot(x, y, zs=z, zdir='z')
        #         d_col_obj = ax.fill_between(x, 0, y, step='pre', alpha=0.01) 
        #         ax.add_collection3d(d_col_obj, zs = z, zdir = 'z')
        #     plot2din3d(x, y, z)
        # # ax.legend(field_amplitudes)
        # # ax.set_xlim(100, 1)
        # ax.set_ylim(0, 1)
        # ax.set_zlim(0, 50)
        # ax.set_xlabel("Time (ps)")
        # ax.set_ylabel("Energy (eV)")
        # ax.set_zlabel("Field amplitude (V/nm)")
        # ax.view_init(elev=20., azim=-35)
        # plt.savefig(i+"_"+j+"_"+"fig.png")

        
       

       




#         """Зависмость значений энергий от поля в последний момент времени"""
#         last_moment_energies = list()
#         legend.append(os.path.basename(directory)+" "+j.lower())
#         for energy_column in data.columns.values[1:]:
#             last_moment_energies.append(float(data.iloc[-1, data.columns.get_loc(energy_column)]))
#         plt.scatter(field_amplitudes, last_moment_energies, s=20)      
# plt.grid()
# plt.xlabel('Field amplitude (V/nm)')
# plt.ylabel('Energy (eV)')
# plt.title('Energy in 500ps time step \n(field amplitude)')
# plt.legend(legend)
# plt.savefig(directory+"/dependence.png")

            
 