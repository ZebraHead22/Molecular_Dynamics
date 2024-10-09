import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def diff_water():
    files = os.listdir(os.getcwd())
    df_water = pd.read_csv(os.getcwd()+'/water.dat',
                           delimiter=' ', index_col=None)
    df_water.rename(columns={'0.0': 'Frequency',
                             '0.0.1': 'Amplitude_water'}, inplace=True)
    df_water.insert(2, 'Amp×104_water', df_water['Amplitude_water']*(10**4))

    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat" and os.path.basename(filename) != 'water':
            print(filename)
            df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                      '0.0.1': 'Amplitude'}, inplace=True)
            df.insert(2, 'Amp×104', df['Amplitude']*(10**4))
            df.insert(3, 'Amp×104_water', df_water['Amp×104_water'])
            df.insert(4, 'Difference',
                      df['Amp×104'] - df_water['Amp×104_water'])
            print(df.head())

            plt.gcf().clear()
            # # Обычные графики спектров
            plt.plot(np.array(df['Frequency']), np.array(
                df['Difference']), linewidth=1)
            plt.ylabel('Spectral Density (a.u.)')
            plt.xlabel('Frequency ($cm^{-1}$)')
            plt.grid()
            plt.savefig(filename+'_under_1000.png')
            plt.show()

diff_water()