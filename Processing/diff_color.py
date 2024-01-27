# Здесь строим кучу спектров из набора дат файлов в одной папке
import os
import re
import matplotlib
import numpy as np
import pandas as pd
from math import sqrt
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


def vertical_graph():
    amplitudes = list()
    frequencies = list()
    am_folder = os.getcwd()
    for address, dirs, files in os.walk(am_folder):
        for name in files:
            filename, file_extension = os.path.splitext(name)
            if file_extension == ".dat":
                df = pd.read_csv(os.path.join(address, name),
                                 delimiter=' ', index_col=None)
                df.rename(columns={'0.0': 'Frequency',
                                   '0.0.1': 'Amplitude'}, inplace=True)
                frequency = re.search(r'\d+', str(os.path.basename(filename)))
                frequency = frequency.group(0)
                closest_value_min = df.iloc[(
                    df['Frequency']-float(int(frequency)-20)).abs().argsort()[:1]].index.tolist()
                closest_value_max = df.iloc[(
                    df['Frequency']-float(int(frequency)+20)).abs().argsort()[:1]].index.tolist()
                max_amplitude = df.loc[closest_value_min[0]:closest_value_max[0], 'Amplitude'].max()
                max_amplitude_frequency = df.loc[df['Amplitude']
                                                 == max_amplitude, 'Frequency'].values[0]
                amplitudes.append(max_amplitude)
                frequencies.append(max_amplitude_frequency)
    plt.gcf().clear()
    plt.scatter(amplitudes, frequencies, c='white',
                s=1, linewidths=5, edgecolors='red')
    plt.grid()
    plt.xlabel('Spectral density (a.u.)')
    plt.ylabel('Frequency ($cm^{-1}$)')
    plt.ylim(0, 1000)
    # thresh = float(min(amplitudes))/float(sqrt(2)/2)
    # plt.xlim(thresh, 0.1)
    # plt.show()
    plt.savefig(am_folder + '/vertical_gly.png')

