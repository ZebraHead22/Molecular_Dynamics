import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

def one_experiment():
    '''
    Здесь считаем зависимость амплитуды пика от амплитуды поля
    '''
    folders = os.listdir(os.getcwd())
    field_amplitudes = [0.0435,	0.087,	0.1305,	0.174,	0.2175,
                    0.261,	0.3045,	0.348,	0.3915,	0.435,	0.4785,	0.522, 0.5655, 0.609]
    for folder in folders:
        print(folder)
        if os.path.isdir(folder) == True:
            dirPath = os.getcwd()+'/'+folder
            files = os.listdir(dirPath)
            d = dict()
            for i in files:
                filename, file_extension = os.path.splitext(dirPath+'/'+i)
                if file_extension == ".dat":
                    df = pd.read_csv(
                        dirPath+'/'+i, delimiter=' ', index_col=None)
                    df.rename(columns={'0.0': 'Frequency',
                              '0.0.1': 'Amplitude'}, inplace=True)
                    max_amp = df['Amplitude'].where(df['Frequency'] > 3000).max()
                    field_amp  = os.path.basename(filename)[0] + '.' + os.path.basename(filename)[1]
                    d[float(field_amp)] = float(max_amp)
            d = dict(sorted(d.items()))
            print(d)
            # x = np.array(list(d.keys()))
            x = np.array(field_amplitudes)
            y = np.array(list(d.values()))
            X_Y_Spline = make_interp_spline(x, y)
            X_ = np.linspace(x.min(), x.max(), 500)
            Y_ = X_Y_Spline(X_)
            plt.plot(X_, Y_,  linewidth=2)
            plt.ylabel('Амплитуда резонансного пика, отн.ед.')
            plt.xlabel('Амплитуда поля (В/нм)')
            
    
    # plt.xlim([0, 1.6])
    folders = folders[1:]
    folders = [x.replace('_', '=') for x in folders]
    plt.legend(folders)
    plt.grid()
    # plt.show()
    plt.savefig(os.getcwd() + '/dependence.png')


one_experiment()
