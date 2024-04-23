import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
'''
Rotate electric field article
Create MaxSpectralDensity(N) - def family + max SD of pAla18 with p = {1..5}
Create MaxSpectralDensity(p) - def maxSpecDen
'''
def family():
    main_df = pd.DataFrame()
    markers = iter(['s', 'o'])

    for folder in os.listdir(os.getcwd()):
        if os.path.isdir(folder):
            print(f"Folder is {folder}")
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
            print(sorted_dict)

            main_df['N'] = list(sorted_dict.keys())
            main_df[str(folder)] = list(sorted_dict.values())

    main_df['average'] = main_df[['plus_circle_wave', 'minus_circle_wave']].mean(axis=1)
    main_df = main_df.drop([17])
    print(main_df.head())

    # max_values = [1.62, 1.95, 4.14, 5.36, 6.12, 10.74, 14.44]
    max_values = [1.62, 1.95, 4.14, 5.36, 6.12]
    # captions = ['I', 'II', 'III', 'IV', 'V', 'X', 'XV']
    captions = ['I', 'II', 'III', 'IV', 'V']

    for max_value in max_values:
        plt.plot(18, max_value, '*', c='red')
    for i, txt in enumerate(captions):
        plt.annotate(txt, (18, max_values[i]), xytext=(18.3, max_values[i]-0.1), fontsize=10)

    plt.scatter(np.array(main_df['N'].tolist()), np.array(main_df['flat_wave'].tolist()), s=25, c='black', marker='s') # Scatter Plot
    a = np.polyfit(np.log(np.array(list(main_df['N'].tolist()))), np.array(list(main_df['flat_wave'].tolist())), 1)  # Approximation coefficients
    y = a[0] * np.log(np.array(list(main_df['N'].tolist()))) + a[1]  # Approximation
    plt.plot(np.array(list(main_df['N'].tolist())), y, 'k--', lw=1)  # Approximate plot
    plt.annotate('2', (0.9, 1.5), fontsize=14)

    plt.scatter(np.array(main_df['N'].tolist()), np.array(main_df['average'].tolist()), s=25, c='black', marker='o') # Scatter Plot
    a = np.polyfit(np.log(np.array(list(main_df['N'].tolist()))), np.array(list(main_df['average'].tolist())), 1)  # Approximation coefficients
    y = a[0] * np.log(np.array(list(main_df['N'].tolist()))) + a[1]  # Approximation
    plt.plot(np.array(list(main_df['N'].tolist())),y, 'k--', lw=1)  # Approximate plot
    plt.annotate('1', (0.9, 4.4), fontsize=14)

    plt.grid()
    plt.xticks(np.arange(0, 22, 2))
    plt.xlabel('N')
    plt.ylabel('Max Apmlitude (a.u. ×$10^{2}$)')
    # plt.show()
    plt.savefig(os.getcwd()+'/family.png', dpi=300)


def maxSpecDen():
    folders = list()
    markers = iter(['s', 'o', '*', 'p',  'v', '2', 'D'])
    for folder in os.listdir(os.getcwd()):
        if os.path.isdir(folder):
            folders.append(int(folder))
            dots = {}
            for data_file in os.listdir(os.getcwd()+'/'+folder):
                file_name, file_extension = os.path.splitext(
                    os.getcwd()+'/'+folder+'/'+data_file)
                if file_extension == '.dat':
                    # print(data_file)
                    dot = re.search(r'\d+', str(os.path.basename(file_name)))
                    dot = dot.group(0)
                    data_df = pd.read_csv(
                        os.getcwd()+'/'+folder+'/'+data_file, sep=' ', index_col=None)
                    data_df.rename(
                        columns={'0.0': 'Freq', '0.0.1': 'Amp'}, inplace=True)
                    closest_value_min = data_df.iloc[(
                        data_df['Freq']-float(int(folder)-1000)).abs().argsort()[:1]].index.tolist()
                    closest_value_max = data_df.iloc[(
                        data_df['Freq']-float(int(folder)+1000)).abs().argsort()[:1]].index.tolist()
                    max_amplitude = data_df.loc[closest_value_min[0]: closest_value_max[0], 'Amp'].max()
                    # max_amplitude_frequency = data_df.loc[data_df['Amplitude']
                    #                                     == max_amplitude, 'Frequency'].values[0]
                    dots[int(dot)] = round(max_amplitude*100, 2)
                    
            sorted_list = sorted(dots.items())
            sorted_dict = {}
            for key, value in sorted_list:
                sorted_dict[key] = value
            print(folder, sorted_dict)

            plt.scatter(np.array(list(sorted_dict.keys())), np.array(list(sorted_dict.values())), s=25, c='black', marker=next(markers)) # Scatter Plot
            # plt.plot(np.array(list(sorted_dict.keys())), np.array(list(sorted_dict.values()))) # Scatter Plot
            
            a = np.polyfit(np.log(np.array(list(sorted_dict.keys()))), np.array(list(sorted_dict.values())), 1)  # Approximation coefficients
            y = a[0] * 0.23 * (np.array(list(sorted_dict.keys()))) + a[1]  # Approximation
            x = np.array(list(sorted_dict.keys()))
            plt.plot(x[1:], y[1:], 'k--', lw=1)  # Approximate plot
            plt.annotate(str(folder), (15.6, max(list(sorted_dict.values()))), fontsize=12)   
           
    plt.grid()
    plt.xticks(np.arange(0, 20, 2))
    plt.xlabel('p')
    plt.ylabel('Max Amplitude (a.u. ×$10^{2}$)')
    # plt.legend(folders)
    # plt.show()
    plt.savefig(os.getcwd()+'/maxSpecDen(p)_all.png', dpi=600)

# maxSpecDen()    
family()    