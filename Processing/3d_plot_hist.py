import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

legend = list()
markers = iter(['+', 'o', 's'])
for folder in os.listdir(os.getcwd()):
    if os.path.isdir(folder):
        print(f"Folder is {folder}")
        legend.append(str(folder))
        legend.append(str(folder + " approximation"))
        value_dict = {}
        for data_file in os.listdir(os.getcwd()+'/'+folder):
            file_name, file_extension = os.path.splitext(
                os.getcwd()+'/'+folder+'/'+data_file)
            if file_extension == '.dat':
                dot = re.search(r'\d+', str(os.path.basename(file_name)))
                dot = dot.group(0)
                data_df = pd.read_csv(
                    os.getcwd()+'/'+folder+'/'+data_file, sep=' ', index_col=None)
                data_df.rename(
                    columns={'0.0': 'Freq', '0.0.1': 'Amp'}, inplace=True)
                max_amp_value = round(
                    (data_df['Amp'].max())*(10**2), 2)  # multiply to 100
                value_dict[int(dot)] = max_amp_value

        sorted_list = sorted(value_dict.items())
        sorted_dict = {}
        for key, value in sorted_list:
            sorted_dict[key] = value
        print(f"Dots on the graph:\n {sorted_dict}")
        plt.scatter(np.array(list(sorted_dict.keys())), np.array(
            # Scatter Plot
            list(sorted_dict.values())), s=25, c='black', marker=next(markers))
        a = np.polyfit(np.log(np.array(list(sorted_dict.keys()))), np.array(
            list(sorted_dict.values())), 1)  # Approximation coefficients
        y = a[0] * np.log(np.array(list(sorted_dict.keys()))
                          ) + a[1]  # Approximation
        plt.plot(np.array(list(sorted_dict.keys())),
                 y, 'k--', lw=1)  # Approximate plot
    else:
        print(f"{folder} is not a dir")

max_values = [1.62, 1.95, 4.14, 5.36, 6.12]
captions = ['1 mol', '2 mol', '3 mol', '4 mol', '5 mol']

for max_value in max_values:
    plt.plot(18, max_value, '*', c='red')
for i, txt in enumerate(captions):
    plt.annotate(txt, (18, max_values[i]), xytext=(18.3, max_values[i]-0.1))

plt.grid()
plt.xticks(np.arange(0, 22, 2))
plt.xlabel('N')
plt.ylabel('Max Spectral Density (a.u. ×$10^{2}$)')
# plt.show()
plt.savefig(os.getcwd()+'/family.png', dpi=300)
