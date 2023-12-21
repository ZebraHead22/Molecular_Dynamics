# Здесь строим кучу спектров из набора дат файлов в одной папке
import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Функция создает набор графиков спектральных зависимостей


def make_spectres():
    folder = os.getcwd()
    files = os.listdir(os.getcwd())
    # file = open("res_freq.txt", "w")
    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                      '0.0.1': 'Amplitude'}, inplace=True)
            print(df.head())
            # df.iloc[:1900] = np.nan
            # df.iloc[21000:] = np.nan
            dfFreq = np.array(df['Frequency'].tolist())
            dfAmp = np.array(df['Amplitude'].tolist())
            # dfAmp = np.array([x*1000 for x in dfAmp])

            # dfAmpRev = list(1 - i for i in dfAmp) #Вычитаем из единицы
            # file.write(str(os.path.basename(filename)+" - " +
            #            str(df.loc[df['Amplitude'].idxmax(), 'Frequency'])+'\n'))
            plt.gcf().clear()
            # Обычные графики спектров
            plt.plot(dfFreq, dfAmp)
            plt.ylabel('Spectral Density (a.u.)')
            plt.xlabel('Frequency ($cm^{-1}$)')
            # plt.xlim(-300, 6300)
            # plt.ylim(0.75, 0.9)
            plt.grid()
            plt.savefig(filename+'.png')
    # file.close()

# Функция строит один спектр


def one_spectrum():
    strings = []
    x_samples = []
    y_samples = []
    wv = []
    files = os.listdir(os.getcwd())
    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                      '0.0.1': 'Amplitude'}, inplace=True)

            frequency = re.search(r'\d+', str(os.path.basename(filename)))
            print(filename)
            print(frequency)
            frequency = frequency.group(0)

            closest_value_min = df.iloc[(
                df['Frequency']-float(int(frequency)-20)).abs().argsort()[:1]].index.tolist()
            closest_value_max = df.iloc[(
                df['Frequency']-float(int(frequency)+20)).abs().argsort()[:1]].index.tolist()
            max_amplitude = df.loc[closest_value_min[0]
                : closest_value_max[0], 'Amplitude'].max()
            max_amplitude_frequency = df.loc[df['Amplitude']
                                             == max_amplitude, 'Frequency']
            x_samples.append(max_amplitude_frequency)
            y_samples.append(max_amplitude)

    x_samples = [float(x) for x in x_samples]
    for i in x_samples:
        wv.append((1/i)*10**4)

    plt.gcf().clear()
    plt.stem(np.array(wv), np.array(y_samples))
    plt.ylabel('Spectral Density (a.u.)')
    plt.xlabel('Wavelenght ($\mu$m)')
    plt.grid()
    plt.savefig("dep.png")

    # x_samples.sort()  # Чекнуть решение струн
    # for i in x_samples:
    #     d = float(float(i)/x_samples[0])
    #     strings.append(round(d))
    # strings.sort()
    # print(strings)
    # print(strings)
    # plt.stem(np.array(wv), np.array(strings))
    # plt.ylabel('Level (a.u.)')
    # plt.xlabel('Frequency ($cm^{-1}$)')
    # plt.grid()
    # plt.show()


def ir_spectres():
    all_data = pd.DataFrame()

    folder = os.getcwd()
    files = os.listdir(folder)
    for file in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+file)
        if file_extension == ".dpt":
            df = pd.read_csv(os.getcwd()+'/'+file, delimiter=',', index_col=None, header=None)
            df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                      1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
            all_data.insert(0, 'Frequency_'+str(os.path.basename(filename)), df['Frequency_'+str(os.path.basename(filename))])
            all_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)), df['Amplitude_'+str(os.path.basename(filename))])
    

    # all_data.iloc[:1900] = np.nan
    # all_data.iloc[21000:] = np.nan

    plt.plot(all_data["Frequency_Valine"], all_data["Amplitude_Valine"], c = 'black')
    plt.plot(all_data["Frequency_Glycine"], all_data["Amplitude_Glycine"], c = 'royalblue')
    plt.plot(all_data["Frequency_Alanine"], all_data["Amplitude_Alanine"], c = 'red')
    plt.plot(all_data["Frequency_Tryptophan"], all_data["Amplitude_Tryptophan"], c = 'green')
    plt.legend(["Valine", "Glycine", "Alanine", "Tryptophan"])
    plt.grid()
    plt.xlim(-300, 6300)
    plt.ylim(0, 1)
    plt.xlabel("Frequency ($cm^{-1}$)")
    plt.ylabel("Energy (a.u.)")
    # plt.show()
    plt.savefig(folder + "/result.png")



make_spectres()