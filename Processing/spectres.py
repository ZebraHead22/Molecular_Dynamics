#Здесь строим кучу спектров из набора дат файлов в одной папке

import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def makeSpectres():
    folder = os.getcwd()
    files = os.listdir(os.getcwd())
    file= open("res_freq.txt", "w")
    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
            df.rename(columns = {'0.0' : 'Frequency', '0.0.1' : 'Amplitude'}, inplace = True)
            dfFreq = np.array(df['Frequency'].tolist())
            dfAmp = np.array(df['Amplitude'].tolist())
            file.write(str(os.path.basename(filename)+" - " +str(df.loc[df['Amplitude'].idxmax(), 'Frequency'])+'\n'))
            plt.gcf().clear()
            plt.plot(dfFreq, dfAmp)
            plt.ylabel('Spectral Density (a.u.)')
            plt.xlabel('Frequency ($cm^{-1}$)')
            plt.grid()
            plt.savefig(filename+'.png')
    file.close()

def oneSpectrum():
    xSamp = []
    ySamp = []
    wv = []
    folder = os.getcwd()
    files = os.listdir(os.getcwd())
    file= open("res_freq.txt", "w")
    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
            df.rename(columns = {'0.0' : 'Frequency', '0.0.1' : 'Amplitude'}, inplace = True)
            xSamp.append(str(df.loc[df['Amplitude'].idxmax(), 'Frequency']))
            ySamp.append(df['Amplitude'].max())
 
    xSamp = [float(x) for x in xSamp]
    for i in xSamp:
        wv.append((1/i)*10**6)

    plt.gcf().clear()
    plt.stem(np.array(wv), np.array(ySamp))
    plt.ylabel('Spectral Density (a.u.)')
    plt.xlabel('Wavelenght ($\mu$m)')
    plt.grid()
    plt.show()

oneSpectrum()    
