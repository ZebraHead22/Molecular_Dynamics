import os
import re
import numpy as np
import scipy as sc
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt

menu_options = {    # Делаем оьъекты меню
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
        plt.plot(x_samples, y_samples, c='darkblue', linewidth = 1)
        plt.vlines(fieldtime, 0, 600, color='r')
        plt.vlines(int(max(x_samples)-fieldtime), 0, 600, color='r')
        plt.ylabel('Square dipole moment (D$^{2}$)')
        plt.xlabel('Time (ps)')
        plt.ylim([0, 600])
        # plt.xlim([0, float(fieldtime)+float(field_realization)+10])
        plt.xlim([0, 110])
        plt.grid()
        plt.savefig(filename+axis+'.png')

        # # Если нет поля, рисовать без полос
        # if os.path.basename(filename) == "dipole_00":
        #     plt.scatter(x_samples, y_samples, c='darkblue', s=7)
        #     plt.ylabel('Dipole moment (D)')
        #     plt.xlabel('Time (ps)')
        #     plt.ylim([-20, 20])
        #     plt.grid()
        #     plt.savefig(filename+axis+'.png')
        # # Если поле есть, отрисовывать полосы
        # else:
        #     plt.scatter(x_samples, y_samples, c='darkblue', s=10)
        #     plt.vlines(fieldtime, int(min(y_samples))-20,
        #                int(max(y_samples))+10, color='r')
        #     plt.vlines(int(max(x_samples)-fieldtime), int(min(y_samples)) -
        #                20, int(max(y_samples))+10, color='r')
        #     plt.ylabel('Dipole moment (D)')
        #     plt.xlabel('Time (ps)')
        #     plt.ylim([int(min(y_samples))-10, int(max(y_samples))+10])
        #     plt.grid()
        #     plt.savefig(filename+axis+'.png')
        
