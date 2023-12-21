import os
import re
import numpy as np
import scipy as sc
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib as mpl

average = list()
field_dict = dict()

for address, dirs, names in os.walk(os.getcwd()):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            a = re.search(r'\d+', name)
            a = int(a.group(0))
            print(name)
            df = pd.read_csv(os.getcwd() + '/' + name, sep=' ')
            df.dropna(how='all', axis=1, inplace=True)
            df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                    'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
            if a == 0:
                df.insert(1, "Time", (df['frame'] * 5 / 1000))
            else:
                df.insert(1, "Time", (df['frame'] * 5 / 1000))

            #Создаем словарь {freq:average D}
            field_dict[int(a)] = float(df['|dip|'].mean())
            #Построение графиков АВР ДМ и сохранение в файл
            plt.gcf().clear()
            plt.plot(np.array(df["Time"].tolist()), np.array(df["dip_x"].tolist()), color='#EB9700', linewidth=2)
            plt.plot(np.array(df["Time"].tolist()), np.array(df["dip_y"].tolist()), color='#EB00A2', linewidth=2)
            plt.plot(np.array(df["Time"].tolist()), np.array(df["dip_z"].tolist()), color='#1EEB00', linewidth=2)
            plt.plot(np.array(df["Time"].tolist()), np.array(df["|dip|"].tolist()), color='#006AEB', linewidth=2)
            plt.legend(["dip_x", "dip_y", "dip_z", "|dip|"])
            plt.ylabel('Dipole Moment (D)')  
            plt.xlabel('Time (ps)')
            plt.grid()
            plt.savefig(os.getcwd() + "/" + 'dipoleplot' + str(a) + ".png")
            print("Ready" + " " + name)

#Обработка словаря
field_dict = dict(sorted(field_dict.items())) #Сортировка
fields = list(field_dict.keys()) 
fields[0] = 'No Field'
fields = [str(x) for x in fields]
dipoles = list(field_dict.values())
#График-гистограмма среднего |ДМ| от частоты поля
plt.gcf().clear()
plt.bar(fields, dipoles, color='darkcyan', width=0.3)
plt.ylabel('Average Dipole Moment (D)')  
plt.xlabel('External Electric Field ($cm^{-1}$)')
plt.grid()
plt.savefig(os.getcwd() + "/" + 'histogram' + ".png")
print("Ready")
