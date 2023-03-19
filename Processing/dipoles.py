import os
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy import signal

def main():
    menu_options = {
        1: 'Dipole X',
        2: 'Dipole y',
        3: 'Dipole z',
        4: 'Dipole ABS',
        5: 'Exit',
    }

    def print_menu():
        for key in menu_options.keys():
            print (key, '--', menu_options[key] )

    print_menu()

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

    timeframe = input("Укажите время разделения фреймов (фс): ")
    timeframe = float(float(timeframe) * (10 ** -15))
    fieldtime = int(input("Укажите время действия поля (пс): "))
    # fieldtime = float(float(fieldtime) * (10 ** -12))

    files = os.listdir(os.getcwd())
    dirPath = os.getcwd()
    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            df = pd.read_csv(dirPath+'/'+i, sep = ' ')
            df.dropna(how='all', axis=1, inplace=True)
            df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': 'dip_abs'}, inplace=True)
            print
            df.insert(1, "Time", (df['frame'] * timeframe)*10**12)
            # print(df.head(-1))
            yList = np.array(df[axis])
            xList = np.array(df["Time"])
            plt.gcf().clear()
            plt.scatter(xList, yList, c = 'darkblue', s = 15)
            plt.vlines(fieldtime, -50, 50, color = 'r')
            plt.ylabel('Dipole moment (D)')
            plt.xlabel('Time (ps)')
            plt.ylim([-20, 20])
            plt.grid()
            plt.savefig(filename+axis+'.png')

def ampDependence():
    maxValues = []
    files = os.listdir(os.getcwd())
    dirPath = os.getcwd()
    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            df = pd.read_csv(dirPath+'/'+i, sep = ' ')
            df.dropna(how='all', axis=1, inplace=True)
            df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': 'dip_abs'}, inplace=True)
            maxValues.append(df['dip_abs'].max())
    maxValues = np.array(maxValues)
    amplitudes = np.array([*range(0, 13, 1)])
    plt.gcf().clear()
    plt.scatter(amplitudes, maxValues, c = 'darkblue', s = 15)
    plt.ylabel('Dipole moment (D)')
    plt.xlabel('External field amplitude (kcal/mol$\cdot$$\AA$$\cdot$e)')
    plt.grid()
    plt.savefig('dependence_ABS.png')

main()
# ampDependence()