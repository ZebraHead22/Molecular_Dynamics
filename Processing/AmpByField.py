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
    field_amplitudes = [0.0435,	0.087,	0.1305,	0.174,	0.2175,
                    0.261,	0.3045,	0.348,	0.3915,	0.435,	0.4785,	0.522, 0.5655, 0.609]
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

            x = np.array(field_amplitudes)
            y = np.array(list(d.values()))
            plt.scatter(x, y, marker=next(markers), c='black', label=str(folder.replace('_', '=')))

            popt, pcov = curve_fit(func, x, y, bounds=(0, [100, 100, 100]))
            print(popt)
            plt.plot(x, func(x, *popt), 'r--')
            

            # popt, pcov = curve_fit(model_f, x, y, p0=[1,1,1])
            # a_opt, b_opt, c_opt = popt
            # x_model = np.linspace(min(x), max(y), 100)
            # y_model = model_f(x_model, a_opt, b_opt, c_opt) 
            # plt.plot(x_model, y_model, color='r')

    plt.legend()
    plt.grid()
    plt.ylabel('Амплитуда резонансного пика, отн.ед.')
    plt.xlabel('Амплитуда поля (В/нм)')
    # plt.show()
    plt.savefig(os.getcwd() + '/dependence.png')


one_experiment()
