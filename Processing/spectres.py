# Здесь строим кучу спектров из набора дат файлов в одной папке
import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Функция создает набор графиков спектральных зависимостей
def make_spectres():
    files = os.listdir(os.getcwd())
    file = open("frequencies.txt", "w")
    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat":
            field_freq = re.search(r'\d{3,}', str(os.path.basename(filename)))
            if field_freq:
                field_freq = field_freq.group(0)
            else:
                field_freq = 'no_field'

            df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                      '0.0.1': 'Amplitude'}, inplace=True)
            df.insert(2, 'Amp×104', df['Amplitude']*(10**4))

            # df.iloc[:1050] = np.nan
            # df.iloc[45000:] = np.nan
            # df = df[df['Frequency'] > 10]
            # df = df.drop(df[df['Frequency'] > 1000].index)

            dfFreq = np.array(df['Frequency'].tolist())
            dfAmp = np.array(df['Amp×104'].tolist())
            # dfAmpRev = list(1 - i for i in dfAmp) #Вычитаем из единицы

            # Ищем точку максимума на резонансной частоте
            G = 50
            max_amp_freq = df.loc[df['Amplitude'].where((df['Frequency'] < (int(
                field_freq) + G)) & (df['Frequency'] > (int(field_freq) - G))).idxmax(), 'Frequency']
            max_amp = df.loc[df['Amplitude'].where((df['Frequency'] < (int(
                field_freq) + G)) & (df['Frequency'] > (int(field_freq) - G))).idxmax(), 'Amp×104']
            
            # Ищем точку максимума без учета частоты поля
            # max_amp_freq = df.loc[df['Amplitude'].idxmax(), 'Frequency']
            # max_amp = df.loc[df['Amplitude'].idxmax(), 'Amp×104']

            mess = 'Field ' + \
                str(field_freq) + " : " + str(max_amp_freq) + \
                ' - ' + str(max_amp) + '\n'

            print(mess)
            # Пишем пики в файл
            file.write(mess)
            # Строим графики
            plt.gcf().clear()
            plt.plot(dfFreq, dfAmp, linewidth=1, c='black')  # Обычный графики
            plt.grid()
            plt.xlabel('Frequency, $cm^{-1}$')
            plt.ylabel('Spectral Density (a.u. ×$10^{4}$)')
            # plt.xlabel('Частота, $cm^{-1}$')
            # plt.ylabel('Амплитуда, отн.ед. ×$10^{4}$')
            # plt.savefig(filename+'_main.png')
            plt.xlim(-50, 1050)
            plt.ylim(-max_amp*0.05, max_amp + max_amp*0.1)
            # plt.savefig(filename+'_u1k.png')

            # Оконные автоматические графики
            D = 10  # Смещение для окон и подписей
            plt.xlim(float(max_amp_freq) - D, float(max_amp_freq) + D)
            plt.ylim(-max_amp*0.03, max_amp + max_amp*0.5)
            plt.annotate(str(round(max_amp_freq, 2)), xy=(float(max_amp_freq), float(max_amp)), xytext=(float(
                max_amp_freq) + 0.5*D, float(max_amp) + float(max_amp)*0.05), arrowprops=dict(facecolor='red', shrink=0.05), fontsize=14)
            # plt.savefig(filename+'_window.png')
    file.close()

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
                df = pd.read_csv(os.getcwd()+'/'+file,
                                 delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                                   1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                trp_data.insert(0, 'Frequency_'+str(os.path.basename(filename)),
                                df['Frequency_'+str(os.path.basename(filename))])
                trp_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)),
                                df['Amplitude_'+str(os.path.basename(filename))])
            elif os.path.basename(filename) == 'Alanine':
                df = pd.read_csv(os.getcwd()+'/'+file,
                                 delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                                   1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                ala_data.insert(0, 'Frequency_'+str(os.path.basename(filename)),
                                df['Frequency_'+str(os.path.basename(filename))])
                ala_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)),
                                df['Amplitude_'+str(os.path.basename(filename))])
            elif os.path.basename(filename) == 'Glycine':
                df = pd.read_csv(os.getcwd()+'/'+file,
                                 delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                                   1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                gly_data.insert(0, 'Frequency_'+str(os.path.basename(filename)),
                                df['Frequency_'+str(os.path.basename(filename))])
                gly_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)),
                                df['Amplitude_'+str(os.path.basename(filename))])
            else:
                df = pd.read_csv(os.getcwd()+'/'+file,
                                 delimiter=',', index_col=None, header=None)
                df.rename(columns={0: 'Frequency_'+str(os.path.basename(filename)),
                                   1: 'Amplitude_'+str(os.path.basename(filename))}, inplace=True)
                val_data.insert(0, 'Frequency_'+str(os.path.basename(filename)),
                                df['Frequency_'+str(os.path.basename(filename))])
                val_data.insert(1, 'Amplitude_'+str(os.path.basename(filename)),
                                df['Amplitude_'+str(os.path.basename(filename))])

    trp_data = trp_data.loc[(trp_data['Frequency_Tryptophan'] >= 500) & (
        trp_data['Frequency_Tryptophan'] <= 5000)]
    ala_data = ala_data.loc[(ala_data['Frequency_Alanine'] >= 500) & (
        ala_data['Frequency_Alanine'] <= 5000)]
    val_data = val_data.loc[(val_data['Frequency_Valine'] >= 500) & (
        val_data['Frequency_Valine'] <= 5000)]
    gly_data = gly_data.loc[(gly_data['Frequency_Glycine'] >= 500) & (
        gly_data['Frequency_Glycine'] <= 5000)]

    plt.plot(val_data["Frequency_Valine"], val_data["Amplitude_Valine"])
    plt.plot(gly_data["Frequency_Glycine"], gly_data["Amplitude_Glycine"])
    plt.plot(ala_data["Frequency_Alanine"], ala_data["Amplitude_Alanine"])
    plt.plot(trp_data["Frequency_Tryptophan"],
             trp_data["Amplitude_Tryptophan"])
    plt.legend(["Valine", "Glycine", "Alanine", "Tryptophan"])
    plt.grid()
    plt.xlim(-300, 6300)
    plt.ylim(0, 1)
    plt.xlabel("Frequency ($cm^{-1}$)")
    plt.ylabel("Energy (a.u.)")
    plt.savefig(folder + "/result.png")

def diff_water():
    files = os.listdir(os.getcwd())
    df_water = pd.read_csv(os.getcwd()+'/water.dat',
                           delimiter=' ', index_col=None)
    df_water.rename(columns={'0.0': 'Frequency',
                             '0.0.1': 'Amplitude_water'}, inplace=True)
    df_water.insert(2, 'Amp×104_water', df_water['Amplitude_water']*(10**4))

    for i in files:
        filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
        if file_extension == ".dat" and os.path.basename(filename) != 'water':
            print(filename)
            df = pd.read_csv(os.getcwd()+'/'+i, delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                      '0.0.1': 'Amplitude'}, inplace=True)
            df.insert(2, 'Amp×104', df['Amplitude']*(10**4))
            df.insert(3, 'Amp×104_water', df_water['Amp×104_water'])
            df.insert(4, 'Difference',
                      df['Amp×104'] - df_water['Amp×104_water'])
            print(df.head())

            plt.gcf().clear()
            # # Обычные графики спектров
            plt.plot(np.array(df['Frequency']), np.array(
                df['Difference']), linewidth=1)
            plt.ylabel('Spectral Density (a.u.)')
            plt.xlabel('Frequency ($cm^{-1}$)')
            # plt.xlim(0, 1000)
            # plt.ylim(0, 2)
            plt.grid()
            # plt.title(str(os.path.basename(filename)))
            # plt.savefig(filename+'_under_1000.png')
            plt.show()


# diff_water()
make_spectres()
