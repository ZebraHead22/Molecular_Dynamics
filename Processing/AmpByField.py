import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

directory = os.getcwd()
folders = os.listdir()
allData = pd.DataFrame()
#Get first folder name
firstFolder = folders[0]
tsDir = directory+'/'+firstFolder
#Take frequency for all data
ts = pd.read_csv(tsDir + '/spec.dat', delimiter=' ', index_col=None)
ts.rename(columns = {'0.0' : 'Freq', '0.0.1' : 'Amplitude'}, inplace = True)
#Write frequency to all Data
allData['Frequency'] = ts['Freq']
#Make array for plot
frequencies = np.array(allData['Frequency'].to_list())
with pd.ExcelWriter(str(directory)+"/"+'result.xlsx') as writer:
    allData.to_excel(writer, sheet_name='first_exp', index=None, index_label=None)
#Read all csv for amplitudes obtain
for folder in folders:
    if os.path.isdir(directory+'/'+folder):
        df = pd.read_csv(folder + '/spec.dat', delimiter=' ', index_col=None)
        df.rename(columns = {'0.0' : 'Freq', '0.0.1' : 'Amplitude'}, inplace = True) 
        allData[os.path.basename(folder)] = df['Amplitude']
        amplitudes = np.array(df['Amplitude'].to_list())
        #Make pictures
        plt.gcf().clear()
        plt.plot(frequencies, amplitudes)
        plt.ylabel('Spectral density (a.u.)')
        plt.xlabel('Frequency ($cm^{-1}$)')
        plt.grid()
        plt.savefig(str(directory)+'/'+str(folder)+'.png')
        #write amplitudes into the ./xlsx file
        with pd.ExcelWriter(str(directory)+"/"+'result.xlsx') as writer:
            allData.to_excel(writer, sheet_name='first_exp', index=None, index_label=None)
               