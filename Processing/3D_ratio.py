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
    # # Ищем все файлы
    # folder = os.getcwd()
    # files = os.listdir(folder)
    # # Формируем столбец частот в общем ДФ
    # all_data = pd.DataFrame()
    # df = pd.read_csv(folder + "/" +files[0], delimiter=',')
    # all_data['Frequency'] = df['Frequency']
    # # Читаем файлы отношений и пишем амплитуды в общий
    # for file in files:
    #     filename, file_extension = os.path.splitext(file)
    #     if file_extension == ".csv":
    #         df = pd.read_csv(folder + "/" +file, delimiter=',')
    #         case = re.search(r'\w+\d+\w+[^_ratio]', filename)
    #         case = case.group(0)
    #         all_data[str(case)] = df['Ratio']
    # print(all_data)
    # all_data.to_csv("ala_ratio.csv", index=False)

    # # Строим график
    # # Set the figure and 3D axes
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Number of bars
    # colors = plt.cm.viridis(np.linspace(0, 1, len(all_data.columns)-1))

    # # Iterate through the columns (molecules) to plot
    # for i, col in enumerate(all_data.columns[1:]):  # skip 'Frequency' column
    #     xs = all_data['Frequency']
    #     ys = np.full_like(all_data[col], i)  # Create y positions for each molecule
    #     zs = np.zeros_like(all_data[col])
    #     dz = all_data[col]
    #     ax.bar3d(xs, ys, zs, 1, 1, dz, color=colors[i], shade=True)

    # # Setting axis labels and ticks
    # ax.set_xlabel('Frequency (cm^-1)')
    # ax.set_ylabel('Molecule')
    # ax.set_zlabel('Ratio (a.u.)')

    # # Set tick labels for y-axis
    # ax.set_yticks(np.arange(len(all_data.columns)-1))
    # ax.set_yticklabels(all_data.columns[1:])  # Exclude 'Frequency' from labels

    # plt.title('3D Colorful Bar Plot')
    # # plt.show()
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Load the CSV file
    file_path = 'trp_ratio.csv'
    data = pd.read_csv(file_path)

    # Extract the necessary data
    frequencies = data['Frequency']
    columns = data.columns[1:]  # Exclude the 'Frequency' column

    # Create lists for the 3D bar plot
    x = []
    y = []
    z = []

    # Prepare the data for 3D bar plotting
    for i, freq in enumerate(frequencies):
        for j, column in enumerate(columns):
            x.append(i)  # Use integer index for x-axis
            y.append(j)  # Use integer index for y-axis
            z.append(data.at[i, column])

    # Create a 3D bar plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define bar dimensions
    dx = np.full(len(x), 0.6)  # width of the bars
    dy = np.full(len(y), 0.6)  # depth of the bars
    dz = z  # height of the bars

    # Normalize z values for color mapping
    norm = plt.Normalize(np.min(z), np.max(z))
    colors = plt.cm.plasma(norm(z))  # Using 'plasma' colormap for a more appealing look

    # Set the font to Times New Roman and increase the font size
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    # Plot the bars with transparency and edge color
    ax.bar3d(x, y, np.zeros_like(z), dx, dy, dz, color=colors, alpha=0.7, edgecolor='k', shade=True)

    # Set axis labels
    ax.set_xlabel('Frequency ($cm^{-1}$)', labelpad=18, fontsize=14)
    ax.set_ylabel('Molecules', labelpad=20, fontsize=14)
    ax.set_zlabel('Amplitude (a.u.)', labelpad=20, fontsize=14)

    # Set x-ticks to frequency values
    ax.set_xticks(range(len(frequencies)))
    ax.set_xticklabels(frequencies, rotation=45, ha='right', fontsize=12)

    # Set y-ticks to column names
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=12)

    # Add a grid for better readability
    ax.grid(True)

    # Show the plot
    plt.show()

   

ratio_graph()