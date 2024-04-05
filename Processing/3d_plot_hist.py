import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

legend = list()
for folder in os.listdir(os.getcwd()):
    print(f"Folder is {folder}")
    legend.append(str(folder))
    value_dict = {}
    for data_file in os.listdir(os.getcwd()+'/'+folder):
        file_name, file_extension = os.path.splitext(os.getcwd()+'/'+folder+'/'+data_file)
        if file_extension == '.dat':
            dot = re.search(r'\d+', str(os.path.basename(file_name)))
            dot = dot.group(0)
            # print(f"Filename is {dot}")
            data_df = pd.read_csv(os.getcwd()+'/'+folder+'/'+data_file, sep=' ', index_col=None)
            data_df.rename(columns={'0.0': 'Freq', '0.0.1': 'Amp'}, inplace=True)
            max_amp_value = round((data_df['Amp'].max())*(10**2), 2) # mul to 100
            # print(f"Max value is {max_amp_value}")
            value_dict[int(dot)] = max_amp_value

    sorted_list = sorted(value_dict.items())

    sorted_dict = {}
    for key, value in sorted_list:
        sorted_dict[key] = value

    print(f"Dots on the graph:\n {sorted_dict}")

    plt.plot(np.array(list(sorted_dict.keys())), np.array(list(sorted_dict.values())))

plt.grid()
plt.legend(legend)
plt.xticks(np.arange(0, 19, 1))
plt.xlabel('N')
plt.ylabel('Max Spectral Density (a.u. ×$10^{2}$)')
# plt.show()
plt.savefig(os.getcwd()+'/family.png')