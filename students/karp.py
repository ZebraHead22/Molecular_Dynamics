import os
import pandas as pd
from matplotlib import pyplot as plt

dir = os.getcwd()
files = os.listdir(dir)
data_frame = pd.DataFrame()
for file in files:
    name, ext = os.path.splitext(file)
    if ext == '.dat':
        data_frame = pd.read_csv(file, sep=' ', index_col=None)
        data_frame.columns = ['Frequency (1/cm)', 'Amplitude (rel.un.)']
print(data_frame)
# data_frame.to_excel('spec_gly.xlsx')

x = data_frame['Frequency (1/cm)']
y = data_frame['Amplitude (rel.un.)']

plt.plot(x, y, linewidth=2)
plt.grid()
plt.xlabel('Frequency (1/cm)')
plt.ylabel('Amplitude (rel.un.)')
plt.savefig('spec_gly.png')
