#Здесь строим кучу спектров из набора дат файлов в одной папке

import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

names = []
mf = []
av = pd.DataFrame()
field_amplutide = []
amplitude = []
list_files = []
folder = os.getcwd()
files = os.listdir(os.getcwd())
for i in files:
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
        df.rename(columns = {'0.0' : 'Frequency', '0.0.1' : 'Amplitude'}, inplace = True)
        # print(df.head())
        dfFreq = np.array(df['Frequency'].tolist())
        dfAmp = np.array(df['Amplitude'].tolist())
        names.append(os.path.basename(filename))
        mf.append(df.loc[df['Amplitude'].idxmax()])
        #mf.append(df['Frequency'].where(df['Amplitude'].max()==df['Amplitude']).dropna()) #Рабочий варик
        #Make png
        plt.gcf().clear()
        plt.plot(dfFreq, dfAmp)
        plt.ylabel('Spectral Density (a.u.)')
        plt.xlabel('Frequency ($cm^{-1}$)')
        plt.grid()
        plt.savefig(filename+'.png')
        print('Picture saved'+' '+filename+'.png')
    
# exp = pd.DataFrame(list(zip(names, mf)), columns =['File', 'Frequency ($cm^{-1}$)'])
# with pd.ExcelWriter('res.xlsx') as writer:
#         exp.to_excel(writer, sheet_name='Dependence', index=None, index_label=None)

with open(folder+'/'+'Freq.txt', 'w') as fp:
    for item in mf, names:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

print(mf)