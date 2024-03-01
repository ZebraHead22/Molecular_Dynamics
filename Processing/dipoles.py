# Здесь считаем полный дипольный момент для нескольких реализаций 

import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.DataFrame()

time = np.linspace(0, 500, 100000)
for i in os.listdir(os.getcwd()):
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        number = re.search(r'\d+', filename)
        number = number.group(0)
        df = pd.read_csv(os.getcwd()+'/'+i, sep=' ')
        df.dropna(how='all', axis=1, inplace=True)
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                  'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
        data['N='+number] = df['|dip|']
    
data = data.reindex(sorted(data.columns), axis=1)
data.insert(0, 'Time (ps)', time)
data = data.set_index('Time (ps)')
data = data.drop(columns=['N=3', 'N=4', 'N=7', 'N=10', 'N=13', 'N=14', 'N=16', 'N=17', 'N=5', 'N=8', 'N=11'], axis=1)
data.plot(linewidth = 1, legend=False)
plt.grid()
plt.ylabel('Dipole Moment (D)')

# plt.show()
plt.savefig(os.getcwd()+'/'+'dipoles_varN.png', dpi=1200)