import os
import re
import numpy as np
import scipy as sc
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib as mpl

menu_options = {    # Делаем объекты меню
    1: 'Dipole X',
    2: 'Dipole y',
    3: 'Dipole z',
    4: 'Dipole ABS',
    5: 'Exit',
}

def print_menu():
    for key in menu_options.keys():
        print(key, '--', menu_options[key])


print_menu()
# Выбор оси
option = int(input('Enter your choice: '))
if option == 1:
    axis = 'dip_x'
    print('Handle option \'Dipole X\'')
elif option == 2:
    axis = 'dip_y'
    print('Handle option \'Dipole Y\'')
elif option == 3:
    axis = 'dip_z'
    print('Handle option \'Dipole Z\'')
elif option == 4:
    axis = 'dip_abs'
    print('Handle option \'Dipole ABS\'')
elif option == 5:
    print('Thanks message before exiting')
    exit()
else:
    print('Invalid option. Please enter a number between 1 and 5.')
# Запросы времен
timeframe = input("Укажите время разделения фреймов (фс): ")
timeframe = float(float(timeframe) * (10 ** -15))
fieldtime = int(input("Укажите время действия поля (пс): "))
# Запрос лимита по Y-оси
y_limit = int(input("Укажите лимиты по Y:  "))
# Тело
files = os.listdir(os.getcwd())
dir_path = os.getcwd()
for i in files:
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        df = pd.read_csv(dir_path+'/'+i, sep=' ')
        df.dropna(how='all', axis=1, inplace=True)
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                  'Unnamed: 6': 'dip_z', 'Unnamed: 8': 'dip_abs'}, inplace=True)
        df.insert(1, "Time", (df['frame'] * timeframe)*10**12)
        y_samples = np.array(df[axis])
        average_value = np.mean(df[axis])
        y_samples = y_samples - average_value  # удаление постоянной составляющей
        y_samples = [x**2 for x in y_samples]
        x_samples = np.array(df["Time"])
        field_realization = re.search(r'\d+', filename)
        field_realization = field_realization.group(0)

        plt.gcf().clear()
        plt.plot(x_samples, y_samples, c='darkblue', linewidth=1)
        # plt.vlines(fieldtime, 0, 600, color='r')
        plt.vlines(int(max(x_samples)-fieldtime), 0, y_limit, color='r')
        # plt.ylabel('Square dipole moment (D$^{2}$)')
        # plt.ylabel('Dipole moment (D\u2013<D>)$^{2}$')
        plt.ylabel('|'+r'$\vec D$(t)-<$\vec D$(t)>'+'|'+r'$^{2}$')
        plt.xlabel('Time (ps)')
        # plt.ylim([0, y_limit])
        # plt.xlim([0, float(fieldtime)+float(field_realization)+10])
        # plt.xlim([200, 201])
        plt.grid()
        plt.savefig(filename+'_'+axis+'.png')

        # dipole = pd.DataFrame(list(zip(x_samples, y_samples)),
        #               columns=['Time (ps)', '(d-<d>)^2ip_x'])

        # with pd.ExcelWriter('dependence.xlsx') as writer:
        #  dipole.to_excel(writer, sheet_name='Dipole', index=None, index_label=None)
