# Здесь строим кучу спектров из набора дат файлов в одной папке
import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def two_classes():
    wv_1 = list()
    wv_2 = list()
    wv_3 = list()
    x_samples_1 = list()
    x_samples_2 = list()
    x_samples_3 = list()
    y_samples_1 = list()
    y_samples_2 = list()
    y_samples_3 = list()
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
            
            if max_amplitude > 0.002:
                x_samples_1.append(max_amplitude_frequency)
                y_samples_1.append(max_amplitude)
            else:
                x_samples_3.append(max_amplitude_frequency)
                y_samples_3.append(max_amplitude)

    x_samples_1 = [float(x) for x in x_samples_1]
    for i in x_samples_1:
        wv_1.append((1/i)*10**4)

    x_samples_3 = [float(x) for x in x_samples_3]
    for i in x_samples_3:
        wv_3.append((1/i)*10**4)

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
    plt.stem(np.array(wv_3), np.array(y_samples_3), 'g', markerfmt='go')

    plt.ylabel('Spectral Density (a.u.)')
    plt.xlabel('Wavelenght ($\mu$m)')
    plt.legend(['New data', 'Literature data', 'Non resonanse'], loc=2)

    plt.grid()
    # plt.show()
    plt.savefig(os.getcwd()+'/'+"TRP.png")


def levels():
    GLY = list()
    FF = list()
    TRP = list()

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

                    max_amplitude_frequency = df.loc[df['Amplitude']
                                                     == max_amplitude, 'Frequency']
                    if folder == "gly":
                        GLY.append(max_amplitude_frequency)
                    elif folder == "ff":
                        FF.append(max_amplitude_frequency)
                    elif folder == "trp":
                        TRP.append(max_amplitude_frequency)  
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

                    max_amplitude_frequency = df.loc[df['Amplitude']
                                                     == max_amplitude, 'Frequency']
                    if folder == "gly":
                        GLY.append(max_amplitude_frequency)
                    elif folder == "ff":
                        FF.append(max_amplitude_frequency)
                    elif folder == "trp":
                        TRP.append(max_amplitude_frequency)  
    for i in np.array(GLY):
        FREQUENCIES_GLY.append(i[0])

    unique = set(FREQUENCIES_GLY)
    for number in unique:
        FREQUENCIES_GLY.append(number)

    for i in np.array(FF):
        FREQUENCIES_FF.append(i[0])

    unique = set(FREQUENCIES_FF)
    for number in unique:
        FREQUENCIES_FF.append(number)

    for i in np.array(TRP):
        FREQUENCIES_TRP.append(i[0])

    unique = set(FREQUENCIES_TRP)
    for number in unique:
        FREQUENCIES_TRP.append(number)
    
    FREQUENCIES_THZ_GLY = [float(x*0.03) for x in FREQUENCIES_GLY]
    FREQUENCIES_THZ_FF = [float(x*0.03) for x in FREQUENCIES_FF]
    FREQUENCIES_THZ_TRP = [float(x*0.03) for x in FREQUENCIES_TRP]
    # ENERGY_GLY = [float(x*4.1) for x in FREQUENCIES_GLY]
    # ENERGY_FF = [float(x*4.1) for x in FREQUENCIES_FF]
    # ENERGY_TRP = [float(x*4.1) for x in FREQUENCIES_TRP]

    plt.gcf().clear()
    fig, ax = plt.subplots()
    ax_e = ax.twinx()
    ax.eventplot(FREQUENCIES_GLY, orientation="vertical", lineoffsets=-1.5, linewidth=0.75, color = "black")
    ax.eventplot(FREQUENCIES_TRP, orientation="vertical", lineoffsets=0, linewidth=0.75, color = 'black')
    ax.eventplot(FREQUENCIES_FF, orientation="vertical", lineoffsets=1.5, linewidth=0.75, color = "black")
    
    # ax.eventplot(FREQUENCIES_THZ_GLY, orientation="vertical", lineoffsets=-1.5, linewidth=0.75, color = "black")
    # ax.eventplot(FREQUENCIES_THZ_TRP, orientation="vertical", lineoffsets=0, linewidth=0.75, color = 'black')
    # ax.eventplot(FREQUENCIES_THZ_FF, orientation="vertical", lineoffsets=1.5, linewidth=0.75, color = "black")

    ax.set_ylabel('Frequency ($cm^{-1}$)')
    ax_e.set_ylabel('Frequency (THz)')
    ax.set_ylim(0, 200)
    ax_e.set_ylim(0, 6)
    # ax_e.set_ylabel('Energy (meV)')
    ax.text(-1.8, -80, 'Glycine                 Tryptophan        Diphenylalanine')
    ax.set_xticks([])

    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
        #   ncol=3, fancybox=True, shadow=True, labels=['Glycine', 'Tryptophan', 'Diphenylalanine'])
    
    fig.savefig(os.getcwd()+'/'+"eveplot_amino_collective.png")



# levels()
levels()