# Здесь строим кучу спектров из набора дат файлов в одной папке
import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def two_classes():
    wv_1 = list()
    wv_2 = list()
    x_samples_1 = list()
    x_samples_2 = list()
    y_samples_1 = list()
    y_samples_2 = list()
    # --------------------------------------------------------------------------------------------
    files_1 = os.listdir(os.getcwd())
    for i in files_1:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                               '0.0.1': 'Amplitude'}, inplace=True)

            frequency = re.search(r'\d+', str(os.path.basename(filename)))
            frequency = frequency.group(0)
            closest_value_min = df.iloc[(
                df['Frequency']-float(int(frequency)-20)).abs().argsort()[:1]].index.tolist()
            closest_value_max = df.iloc[(
                df['Frequency']-float(int(frequency)+20)).abs().argsort()[:1]].index.tolist()
            max_amplitude = df.loc[closest_value_min[0]: closest_value_max[0], 'Amplitude'].max()
            max_amplitude_frequency = df.loc[df['Amplitude']
                                             == max_amplitude, 'Frequency']
            x_samples_1.append(max_amplitude_frequency)
            y_samples_1.append(max_amplitude)

    x_samples_1 = [float(x) for x in x_samples_1]
    for i in x_samples_1:
        wv_1.append((1/i)*10**4)
    # --------------------------------------------------------------------------------------------
    files_2 = os.listdir(os.getcwd()+"/Literature/")
    for i in files_2:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            df = pd.read_csv(os.getcwd()+'/Literature/'+i,
                             delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                               '0.0.1': 'Amplitude'}, inplace=True)

            frequency = re.search(r'\d+', str(os.path.basename(filename)))
            frequency = frequency.group(0)

            closest_value_min = df.iloc[(
                df['Frequency']-float(int(frequency)-20)).abs().argsort()[:1]].index.tolist()
            closest_value_max = df.iloc[(
                df['Frequency']-float(int(frequency)+20)).abs().argsort()[:1]].index.tolist()
            max_amplitude = df.loc[closest_value_min[0]: closest_value_max[0], 'Amplitude'].max()
            max_amplitude_frequency = df.loc[df['Amplitude']
                                             == max_amplitude, 'Frequency']
            x_samples_2.append(max_amplitude_frequency)
            y_samples_2.append(max_amplitude)

    x_samples_2 = [float(x) for x in x_samples_2]
    for i in x_samples_2:
        wv_2.append((1/i)*10**4)
    # # --------------------------------------------------------------------------------------------
    plt.gcf().clear()
    plt.stem(np.array(wv_1), np.array(y_samples_1), 'r', markerfmt='ro')
    plt.stem(np.array(wv_2), np.array(y_samples_2), 'b', markerfmt='bo')

    plt.ylabel('Spectral Density (a.u.)')
    plt.xlabel('Wavelenght ($\mu$m)')
    plt.legend(['New data', 'Literature data'], loc=2)

    plt.grid()
    # plt.show()
    plt.savefig(os.getcwd()+'/'+"dep.png")


def levels():
    FREQUENCIES_GLY = list()
    FREQUENCIES_FF = list()
    FREQUENCIES_TRP = list()

    molecules_path = os.getcwd()
    folders = os.listdir(molecules_path)
    for folder in folders:
        if os.path.isdir(folder) == True:
# --------------------------------------------------------------------------------------------
            files = os.listdir(folder)
            for some_file in files:
                filename, file_extension = os.path.splitext(folder+'/'+some_file)
                if file_extension == ".dat":
                    
                    df = pd.read_csv(
                        folder+'/'+some_file, delimiter=' ', index_col=None)

                    df.rename(columns={'0.0': 'Frequency',
                                       '0.0.1': 'Amplitude'}, inplace=True)

                    frequency = re.search(
                        r'\d+', str(os.path.basename(filename)))
                    frequency = frequency.group(0)

                    closest_value_min = df.iloc[(
                        df['Frequency']-float(int(frequency)-20)).abs().argsort()[:1]].index.tolist()
                    closest_value_max = df.iloc[(
                        df['Frequency']-float(int(frequency)+20)).abs().argsort()[:1]].index.tolist()
                    max_amplitude = df.loc[closest_value_min[0]: closest_value_max[0], 'Amplitude'].max()
                    print(max_amplitude)
                    max_amplitude_frequency = df.loc[df['Amplitude']
                                                     == max_amplitude, 'Frequency']
                    if folder == "gly":
                        FREQUENCIES_GLY.append(max_amplitude_frequency)
                    elif folder == "ff":
                        FREQUENCIES_FF.append(max_amplitude_frequency)
                    elif folder == "trp":
                        FREQUENCIES_TRP.append(max_amplitude_frequency)  
# # --------------------------------------------------------------------------------------------
            literature_files = os.listdir(folder+"/Literature/")
            for lit_file in literature_files:
                filename, file_extension = os.path.splitext(folder+'/Literature/'+lit_file)
                if file_extension == ".dat":

                    df = pd.read_csv(
                        folder+'/'+some_file, delimiter=' ', index_col=None)

                    df.rename(columns={'0.0': 'Frequency',
                                       '0.0.1': 'Amplitude'}, inplace=True)

                    frequency = re.search(
                        r'\d+', str(os.path.basename(filename)))
                    frequency = frequency.group(0)

                    closest_value_min = df.iloc[(
                        df['Frequency']-float(int(frequency)-20)).abs().argsort()[:1]].index.tolist()
                    closest_value_max = df.iloc[(
                        df['Frequency']-float(int(frequency)+20)).abs().argsort()[:1]].index.tolist()
                    max_amplitude = df.loc[closest_value_min[0]: closest_value_max[0], 'Amplitude'].max()
                    print(max_amplitude)
                    max_amplitude_frequency = df.loc[df['Amplitude']
                                                     == max_amplitude, 'Frequency']
                    if folder == "gly":
                        FREQUENCIES_GLY.append(max_amplitude_frequency)
                    elif folder == "ff":
                        FREQUENCIES_FF.append(max_amplitude_frequency)
                    elif folder == "trp":
                        FREQUENCIES_TRP.append(max_amplitude_frequency)  


    FREQUENCIES_GLY = [float(x*0.03) for x in FREQUENCIES_GLY]
    FREQUENCIES_FF = [float(x*0.03) for x in FREQUENCIES_FF]
    FREQUENCIES_TRP = [float(x*0.03) for x in FREQUENCIES_TRP]
    ENERGY_GLY = [float(x*4.1) for x in FREQUENCIES_GLY]
    ENERGY_FF = [float(x*4.1) for x in FREQUENCIES_FF]
    ENERGY_TRP = [float(x*4.1) for x in FREQUENCIES_TRP]

    plt.gcf().clear()
    fig, ax = plt.subplots()
    ax_e = ax.twinx()
    ax.eventplot(FREQUENCIES_GLY, orientation="vertical", lineoffsets=-1.5, linewidth=0.75, color = "black")
    ax.eventplot(FREQUENCIES_TRP, orientation="vertical", lineoffsets=0, linewidth=0.75, color = 'black')
    ax.eventplot(FREQUENCIES_FF, orientation="vertical", lineoffsets=1.5, linewidth=0.75, color = "black")
    
    ax.set_ylabel('Frequency (THz)')
    ax.set_ylim(0, 15)
    ax_e.set_ylim(0, 61.5)
    ax_e.set_ylabel('Energy (meV)')
    ax.text(-1.8, 14, 'Glycine                 Tryptophan        Diphenylalanine')
    ax.set_xticks([])

    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
        #   ncol=3, fancybox=True, shadow=True, labels=['Glycine', 'Tryptophan', 'Diphenylalanine'])
    
    fig.savefig(os.getcwd()+'/'+"eveplot_amino.png")



levels()