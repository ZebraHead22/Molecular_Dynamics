import os
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy import signal

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
    axis = '|dip|'
    print('Handle option \'Dipole ABS\'')
elif option == 5:
    print('Thanks message before exiting')
    exit()
else:
    print('Invalid option. Please enter a number between 1 and 5.')

timeframe = input("Укажите время разделения фреймов: ")
timeframe = float(float(timeframe) * (10 ** -15))

files = os.listdir(os.getcwd())
dirPath = os.getcwd()
for i in files:
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        df = pd.read_csv(dirPath+'/'+i, sep = ' ')
        df.dropna(how='all', axis=1, inplace=True)
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
        df.insert(1, "Time", (df['frame'] * timeframe)*10**12)
        # print(df.head(-1))
        yList = np.array(df[axis])
        xList = np.array(df["Time"])
        plt.gcf().clear()
        plt.plot(xList, yList)
        plt.ylabel('Dipole moment (D)')
        plt.xlabel('Time (ps)')
        plt.grid()
        plt.savefig(filename+axis+'.png')

