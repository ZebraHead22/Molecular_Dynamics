import os
import re
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline

def ratio():
    # Готовим файл амплитуд в знаменатель
    zero_data = pd.read_csv(
        '/Users/max/Yandex.Disk.localized/NAMD/basic_ak_no_field/trp.dat', delimiter=' ', index_col=None)
    zero_data.rename(columns={'0.0': 'Frequency',
                     '0.0.1': 'Amplitude'}, inplace=True)
    zero_data.insert(2, 'Amp×104', zero_data['Amplitude']*(10**4))
    
    # Формируем отсортированный по частотам список dat файлов
    files = os.listdir(os.getcwd())
    dat_files = list()
    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            dat_files.append(i)
    dat_files = sorted(dat_files, key=lambda x: int((re.findall(r'\d{3,}', x))[0]))
    
    # Поля для заголовков в csv файл
    fields = ['Frequency', 'Ratio']
    mydict = list()
    # Ищем максимальные амплитуды и считаем отношение   
    for file in dat_files:
        field_freq = re.search(r'\d{3,}', str(os.path.basename(file)))
        field_freq = field_freq.group(0)
        df = pd.read_csv(os.getcwd()+'/'+file, delimiter=' ', index_col=None)
        df.rename(columns={'0.0': 'Frequency',
                '0.0.1': 'Amplitude'}, inplace=True)
        df.insert(2, 'Amp×104', df['Amplitude']*(10**4))
        # Ищем точку максимума на резонансной частоте
        G = 50
        max_amp_freq = df.loc[df['Amplitude'].where((df['Frequency'] < (int(
            field_freq) + G)) & (df['Frequency'] > (int(field_freq) - G))).idxmax(), 'Frequency']
        max_amp = df.loc[df['Amplitude'].where((df['Frequency'] < (int(
            field_freq) + G)) & (df['Frequency'] > (int(field_freq) - G))).idxmax(), 'Amp×104']
        max_amp_no_field = zero_data.loc[zero_data['Amplitude'].where((zero_data['Frequency'] < (float(
            max_amp_freq) + 1)) & (zero_data['Frequency'] > (float(max_amp_freq) - 1))).idxmax(), 'Amp×104']
        ratio = max_amp / max_amp_no_field
        ratio = round(ratio, 2)
        mess = {str(fields[0]) : int(field_freq), str(fields[1]) : ratio}
        print(mess)
        mydict.append(mess)

    # Пишем файл
    filename = "../trp_ratio/" + os.path.basename(os.getcwd()) + "_ratio.csv"
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mydict)
# ratio()

def ratio_graph():
    # Ищем все файлы
    folder = os.getcwd()
    files = os.listdir(folder)
    # Формируем столбец частот в общем ДФ
    all_data = pd.DataFrame()
    df = pd.read_csv(folder + "/" +files[0], delimiter=',')
    all_data['Frequency'] = df['Frequency']
    # Читаем файлы отношений и пишем амплитуды в общий
    for file in files:
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".csv":
            df = pd.read_csv(folder + "/" +file, delimiter=',')
            case = re.search(r'\w+\d+\w+[^_ratio]', filename)
            case = case.group(0)
            all_data[str(case)] = df['Ratio']
    print(all_data)
    all_data.to_csv("ala_ratio.csv", index=False)

    # Строим график
    # Set the figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Number of bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_data.columns)-1))

    # Iterate through the columns (molecules) to plot
    for i, col in enumerate(all_data.columns[1:]):  # skip 'Frequency' column
        xs = all_data['Frequency']
        ys = np.full_like(all_data[col], i)  # Create y positions for each molecule
        zs = np.zeros_like(all_data[col])
        dz = all_data[col]
        ax.bar3d(xs, ys, zs, 1, 1, dz, color=colors[i], shade=True)

    # Setting axis labels and ticks
    ax.set_xlabel('Frequency (cm^-1)')
    ax.set_ylabel('Molecule')
    ax.set_zlabel('Ratio (a.u.)')

    # Set tick labels for y-axis
    ax.set_yticks(np.arange(len(all_data.columns)-1))
    ax.set_yticklabels(all_data.columns[1:])  # Exclude 'Frequency' from labels

    plt.title('3D Colorful Bar Plot')
    # plt.show()
    


ratio_graph()