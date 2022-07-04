import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path = '/Volumes/SamsungT5/localNamd/3_part_temperature/gly_3333/vacuum/305/'
df = pd.read_csv(path+'spec.dat', delimiter=' ', index_col=None)
df.rename(columns = {'0.0' : 'Freq', '0.0.1' : 'Amplitude'}, inplace = True) 

frequencies = np.array(df['Freq'].to_list())
amplitudes = np.array(df['Amplitude'].to_list())

plt.plot(frequencies, amplitudes)
plt.ylabel('Amplitude (rel.u.)')
plt.xlabel('Frequency ($cm^{-1}$)')
plt.grid()
plt.show()
# path = os.getcwd()
# plt.savefig(str(path)+'saved_figure.png')