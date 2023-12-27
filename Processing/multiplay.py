import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 33169 points, 15994.34 cm-1
IR_FILE = '/Users/max/Yandex.Disk.localized/Competititons/РНФ/Инфраструктура/DataScience/ИК_data/Val/def_from_1/15582_52_attitude_medium_IR.dpt'
# 89926 points, 5999.2 cm-1
MD_FILE = '/Users/max/Yandex.Disk.localized/NAMD/test/val.dat'

IR_DATA = pd.read_csv(IR_FILE, delimiter=',', index_col=None, header=None)
IR_DATA.rename(columns={0: 'Frequency',
                        1: 'Amplitude'}, inplace=True)
MD_DATA = pd.read_csv(MD_FILE, delimiter=' ', index_col=None)
MD_DATA.rename(columns={'0.0': 'Frequency',
                        '0.0.1': 'Amplitude'}, inplace=True)

IR_DATA["Frequency"] = [round(x) for x in IR_DATA["Frequency"].tolist()]
IR_DATA = IR_DATA.set_index("Frequency")
IR_DATA = IR_DATA.groupby(level='Frequency').mean()
IR_DATA = IR_DATA.drop(IR_DATA.index[6001:])
IR_DATA = IR_DATA.reset_index()  # 5980 points, 5981 cm-1

MD_DATA["Frequency"] = [round(x) for x in MD_DATA["Frequency"].tolist()]
MD_DATA = MD_DATA.set_index("Frequency")
MD_DATA = MD_DATA.groupby(level='Frequency').mean()
MD_DATA = MD_DATA.drop(MD_DATA.index[6030:])
MD_DATA = MD_DATA.reset_index()  # 5980 points, 5980 cm-1

df = pd.DataFrame()
df['Frequency'] = MD_DATA["Frequency"]
df['IR Amplitudes'] = IR_DATA['Amplitude']
df['MD Amplitudes'] = MD_DATA['Amplitude']
df['Correlation'] = (IR_DATA['Amplitude'] * MD_DATA['Amplitude'])*(10**4)

plt.gcf().clear()
plt.plot(df['Frequency'].to_list(), df['Correlation'].tolist(), c = '#2D4354')
plt.grid()
plt.xlabel('Frequency ($cm^{-1}$)')
plt.ylabel('Multiply MD & IR Spectral Density (a.u.)')
plt.title('Valine')
plt.ylim([-0.3, 1])
plt.savefig(os.getcwd()+'/valine.png')
