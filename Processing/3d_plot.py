import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

l=list()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')   
amino_acids = list()
legend = list()
frequencies = list()
energy_type = ['KINETIC', 'POTENTIAL']
field_amplitudes = [0.0435,	0.087,	0.1305,	0.174,	0.2175,
                    0.261,	0.3045,	0.348,	0.3915,	0.435,	0.4785,	0.522]

slice_of_energy = pd.DataFrame()
directory = os.getcwd()

for dir in os.listdir(directory):
    if os.path.isdir(dir) == True:
        frequencies.append(int(dir))
frequencies = sorted(list(set(frequencies)))
print('Frequencies is ', frequencies)  

for address, dirs, names in os.walk(directory):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            amino_acid = re.search(r'^\w{,2}[^\_]', filename)
            amino_acid = amino_acid.group(0)
            amino_acids.append(amino_acid)
amino_acids = sorted(list(set(amino_acids)))
print('Amino acids is ', amino_acids)

for i in amino_acids:
    for j in energy_type:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        out = pd.DataFrame()
        for f in frequencies:
            data = pd.DataFrame()
            files = os.listdir(directory+"/"+str(f)+'/'+i)
            files = sorted(files)
            for file in files:
                filename, file_extension = os.path.splitext(file)
                if file_extension == ".dat":
                    one_file_data = pd.read_csv(
                        directory+"/"+str(f)+'/'+i+"/"+file, delimiter=' ', index_col=None, header=[0])
                    data[(str(filename)+"_"+j)] = round((one_file_data[j])*0.0434/5400, 3) # переводим усл.ед. в эВ
            last_moment_energies = list()
            for energy_column in data.columns.values:
                last_moment_energies.append(
                    float(data.iloc[-1, data.columns.get_loc(energy_column)]))           
            out[i+str(f)+j] = last_moment_energies
        # print(out)
            print('Processing of ', i+' '+str(f)+' '+j) 
            
            X_Y_Spline = make_interp_spline(field_amplitudes, last_moment_energies)
            X_ = np.linspace((np.array(field_amplitudes)).min(), (np.array(field_amplitudes)).max(), 500)
            Y_ = X_Y_Spline(X_)
            # plt.plot(X_, Y_,  linewidth=2)
            ax.plot(X_, Y_, f, zdir='y', marker = 'o', markersize=0.3,  linewidth = 2)
        # ax.set_yticks(frequencies)
        ax.set_title((str(i)).upper()+' '+str(j))
        ax.set_xlabel('Field Amplitude (V/nm)')
        ax.set_ylabel('Field Frequency ($cm^{-1}$)')
        ax.set_zlabel('Amplitude (eV)')
        ax.grid()
        plt.legend([str(x)+'$cm^{-1}$' for x in frequencies])
        # plt.yticks(rotation=45)
        plt.show()
        # ax.figure.savefig(directory+'/'+str(i)+'_'+str(j).lower()+".png")
