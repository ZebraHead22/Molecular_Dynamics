import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

def one_experiment():
    '''
    Здесь считаем зависимость амплитуды пика от амплитуды поля
    '''
    markers = iter(['s', 'o'])
    folders = os.listdir(os.getcwd())
    
    def func(x, a, b, c):
        return a * np.log(b * x) + c

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
            # print(d)
            if folder == 'N_18':
                field_amplitudes = [0.1305,	0.174,	0.2175,
                    0.261,	0.3045,	0.348,	0.3915,	0.435,	0.4785,	0.522, 0.5655]
                x = np.array(field_amplitudes)
            else:
                field_amplitudes = [0.1305,	0.174,	0.2175,
                    0.261,	0.3045,	0.348,	0.3915,	0.435]
                x = np.array(field_amplitudes)
            y = np.array(list(d.values()))
            print(y)
            plt.scatter(x, y, marker=next(markers), c='black', label=str(folder.replace('_', '=')))

            popt, pcov = curve_fit(func, x, y, bounds=(0, [100, 100, 100]))
            # print(popt)
            plt.plot(x, func(x, *popt), 'r--')
            
    plt.scatter([0.0435, 0.087,	0.4785,	0.522, 0.5655], [0.0009087709018414, 0.0016396899418543, 0.0412094721819672, 0.0370683804002949, 0.0316645102820092], marker='s', c='black')
    plt.scatter([0.0435, 0.087], [0.000724186306351, 0.0007593006714712], marker='o', c='black')

    plt.legend()
    plt.grid()
    plt.ylabel('Амплитуда резонансного пика, отн.ед.')
    plt.xlabel('Амплитуда поля (В/нм)')
    # plt.show()
    plt.savefig(os.getcwd() + '/dependence.png')


one_experiment()
