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
            df.insert(2, 'Amp×104', df['Amplitude']*(10**4))
            print(df.head())
            # df.iloc[:1900] = np.nan
            # df.iloc[21000:] = np.nan
            # df = df.loc[(df['Frequency'] >=500 ) & (df['Frequency'] <=5000)]
            dfFreq = np.array(df['Frequency'].tolist())
            dfAmp = np.array(df['Amplitude'].tolist())
            # # dfAmp = [x*10**4 for x in dfAmp]
            # # print(df)
            dfAmp = np.array([x*10000 for x in dfAmp])

            # # dfAmpRev = list(1 - i for i in dfAmp) #Вычитаем из единицы
            # # file.write(str(os.path.basename(filename)+" - " +
            # #            str(df.loc[df['Amplitude'].idxmax(), 'Frequency'])+'\n'))
            # df.to_excel(filename+'.xlsx')
            
            plt.gcf().clear()
            # # Обычные графики спектров
            plt.plot(dfFreq, dfAmp, linewidth=1)
            plt.ylabel('Spectral Density (a.u. ×$10^{4}$)')
            plt.xlabel('Frequency ($cm^{-1}$)')
            # plt.xlim(3200, 3500)
            # plt.ylim(0, 2)
            plt.grid()
            # plt.title(str(os.path.basename(filename)))
            plt.savefig(filename+'.png')
            # plt.show()
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
    # plt.xlabel('Wavelenght ($\mu$m)')
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
    gly_data = pd.DataFrame()
    val_data = pd.DataFrame()
    trp_data = pd.DataFrame()
    ala_data = pd.DataFrame()
    

    folder = os.getcwd()
    files = os.listdir(folder)
    for file in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+file)
        if file_extension == ".dpt":
            if os.path.basename(filename) == 'Tryptophan':
                df = pd.read_csv(os.getcwd()+'/'+file, delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                        1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                trp_data.insert(0, 'Frequency_'+str(os.path.basename(filename)), df['Frequency_'+str(os.path.basename(filename))])
                trp_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)), df['Amplitude_'+str(os.path.basename(filename))])
            elif os.path.basename(filename) == 'Alanine':
                df = pd.read_csv(os.getcwd()+'/'+file, delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                        1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                ala_data.insert(0, 'Frequency_'+str(os.path.basename(filename)), df['Frequency_'+str(os.path.basename(filename))])
                ala_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)), df['Amplitude_'+str(os.path.basename(filename))])
            elif os.path.basename(filename) == 'Glycine':
                df = pd.read_csv(os.getcwd()+'/'+file, delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                        1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                gly_data.insert(0, 'Frequency_'+str(os.path.basename(filename)), df['Frequency_'+str(os.path.basename(filename))])
                gly_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)), df['Amplitude_'+str(os.path.basename(filename))])
            else:
                df = pd.read_csv(os.getcwd()+'/'+file, delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                        1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                val_data.insert(0, 'Frequency_'+str(os.path.basename(filename)), df['Frequency_'+str(os.path.basename(filename))])
                val_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)), df['Amplitude_'+str(os.path.basename(filename))])


    trp_data = trp_data.loc[(trp_data['Frequency_Tryptophan'] >=500) & (trp_data['Frequency_Tryptophan'] <=5000)] 
    ala_data = ala_data.loc[(ala_data['Frequency_Alanine'] >=500) & (ala_data['Frequency_Alanine'] <=5000)] 
    val_data = val_data.loc[(val_data['Frequency_Valine'] >=500) & (val_data['Frequency_Valine'] <=5000)] 
    gly_data = gly_data.loc[(gly_data['Frequency_Glycine'] >=500) & (gly_data['Frequency_Glycine'] <=5000)] 

    plt.plot(val_data["Frequency_Valine"], val_data["Amplitude_Valine"])
    plt.plot(gly_data["Frequency_Glycine"], gly_data["Amplitude_Glycine"])
    plt.plot(ala_data["Frequency_Alanine"], ala_data["Amplitude_Alanine"])
    plt.plot(trp_data["Frequency_Tryptophan"], trp_data["Amplitude_Tryptophan"])
    plt.legend(["Valine", "Glycine", "Alanine", "Tryptophan"])
    plt.grid()
    plt.xlim(-300, 6300)
    plt.ylim(0, 1)
    plt.xlabel("Frequency ($cm^{-1}$)")
    plt.ylabel("Energy (a.u.)")
    plt.savefig(folder + "/result.png")



make_spectres()