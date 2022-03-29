import os
import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack
from pandas import Series
from matplotlib import pyplot as plt

i=1
big_df = pd.DataFrame()
path="d://namd//trpVR//results"
dir=os.chdir(path)
files=os.listdir()
for file in files:
    filename, file_extension = os.path.splitext(str(dir)+"/"+file)
    if file_extension == ".csv":
        df=pd.read_csv(file)
        # big_df['TS']=df['TS']
        big_df['ELECT'+str(i)]=np.array(df['ELECT'].tolist())/4131*0.0434*1000
        i+=1
#average
# mean = big_df.mean(axis=1)
# big_df['AVERAGE']=big_df.mean(axis=1)
#Sum
big_df['AVERAGE']=df.sum(axis=1)
file = 'output_1.csv'
#get times
tsdf=pd.read_csv(file)
times=tsdf['TS']
times = np.array(Series.tolist(times))*(10**(-15))
cutTime=float(times[1]-times[0])
sampleRate=round((float(1/cutTime)))
print(big_df)
energy1=np.array(big_df['ELECT1'].tolist())
window=np.hanning(int(round(len(times))))
y_res1=energy1*window
energies_fft1=sp.fftpack.fft(np.array(y_res1))
energies_psd1=np.abs(energies_fft1)
fftFreq=sp.fftpack.fftfreq(len(energies_fft1), 1/float(sampleRate))
i=fftFreq>0

energy2=np.array(big_df['AVERAGE'].tolist())    
window=np.hanning(int(round(len(times))))
y_res2=energy2*window
energies_fft2=sp.fftpack.fft(np.array(y_res2))
energies_psd2=np.abs(energies_fft2)
fftFreq=sp.fftpack.fftfreq(len(energies_fft2), 1/float(sampleRate))
i=fftFreq>0

# fig = plt.figure()
# ax_1 = fig.add_subplot(1, 2, 1)
# ax_2 = fig.add_subplot(1, 2, 2)
# ax_1.plot(fftFreq[i], 10 * np.log10(energies_psd1[i]), color = 'black', linewidth = 2)
# ax_2.plot(fftFreq[i], 10 * np.log10(energies_psd2[i]), color = 'black', linewidth = 2)
# ax_1.set(title = 'One realization', xticks=[], yticks=[])
# ax_2.set(title = 'Average (10 real.)', xticks=[], yticks=[])
# ax_1.xaxis.get_data_interval()
# ax_1.yaxis.get_data_interval()
# ax_1.margins(0.05)
# ax_2.margins(0.05)
# plt.show()

plt.plot(fftFreq[i], 10 * np.log10(energies_psd2[i]))
plt.show()