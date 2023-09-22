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

amino_acids = list()
legend = list()
energy_type = ['KINETIC', 'POTENTIAL']
field_amplitudes = [0.0435,	0.087,	0.1305,	0.174,	0.2175,
                    0.261,	0.3045,	0.348,	0.3915,	0.435,	0.4785,	0.522]
data = pd.DataFrame()
slice_of_energy = pd.DataFrame()
directory = os.getcwd()
'''
Этот кусок (цикл) ищет папки с АК и пишет их в список
'''
for address, dirs, names in os.walk(directory):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            amino_acid = re.search(r'^\w{,2}[^\_]', filename)
            amino_acid = amino_acid.group(0)
            amino_acids.append(amino_acid)
amino_acids = list(set(amino_acids))
'''
Формирует список .dat файлов в папке    определенной аминокислоты + сортировка
'''
for i in amino_acids:
    directory = os.getcwd()
    directory = directory+"/"+i
    dat_files = list()
    for file in os.listdir(directory):
        dat_files.append(file)
    dat_files.sort()
    '''
    Делаем парсинг и пишем в .xlsx
    '''
    for j in energy_type:
        data = pd.DataFrame()
        for file in dat_files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == ".dat":
                one_data = pd.read_csv(
                    directory+"/"+file, delimiter=' ', index_col=None, header=[0])
                data["TIME"] = (one_data["TS"]) * 0.001
                data[(str(filename)+"_"+j)] = (one_data[j]) * 0.0434/5400
        # print(data.head())
        # data.to_excel(directory+"_"+j+".xlsx")
        '''
        Тут считаются спектры
        '''
        spectre_data = pd.DataFrame()
        for energy_column in data.columns.values[1:]:
            energies_psd = np.abs(sp.fftpack.fft(data[energy_column].tolist()))
            n = len(energies_psd)
            window = sp.signal.windows.gaussian(n, std=5000, sym=False)
            module_w = energies_psd
            b, a = sp.signal.butter(2, 0.05)
            filtered = sp.signal.filtfilt(b, a, module_w)
            fftFreq = sp.fftpack.fftfreq(
                len(data[energy_column].tolist()), 1/10**15)
            z = fftFreq > 0
            reverseCm = 1/((3*(10**10))/(fftFreq[z]))
            spectre_data["FREQUENCY"] = reverseCm
            spectre_data[(str(energy_column))] = filtered[z]
        # print(spectre_data.head())
        # spectre_data.to_excel(directory+"_"+j+"_spectre.xlsx")
        """
        Делаем графики по последнему моменту
        """
        # last_moment_energies = list()
        # legend.append(os.path.basename(directory)+" "+j.lower())
        # for energy_column in data.columns.values[1:]:
        #     last_moment_energies.append(
        #         float(data.iloc[-1, data.columns.get_loc(energy_column)]))
        # plt.scatter(field_amplitudes, last_moment_energies, s=20)
        # plt.grid()
        # plt.xlabel('Field amplitude (V/nm)')
        # plt.ylabel('Energy (eV)')
        # plt.title('Energy in 500ps time step \n(field amplitude)')
        # plt.legend(legend)
        # plt.savefig(directory+"/dependence.png")
        '''
        Делаем срезы в спектрах
        '''
        spectre_data["FREQUENCY"] = [round(x, 2) for x in spectre_data["FREQUENCY"].tolist()]
        spectre_data = spectre_data.set_index("FREQUENCY")
        slice_of_energy[str(i)+'_'+str(j)] = spectre_data.loc[3333.33].to_list()
slice_of_energy.insert(0, "FIELD_AMPLITUDES", field_amplitudes)
plot = slice_of_energy.plot(x='FIELD_AMPLITUDES', title="DataFrame Plot")
plt.show()
