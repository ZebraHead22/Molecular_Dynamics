import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

# l=list()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')   
# amino_acids = list()
# legend = list()
# frequencies = list()
# energy_type = ['KINETIC', 'POTENTIAL']
# field_amplitudes = [0.0435,	0.087,	0.1305,	0.174,	0.2175,
#                     0.261,	0.3045,	0.348,	0.3915,	0.435,	0.4785,	0.522]
# norm_freq = [0 , 100, 200, 300, 400, 500]

# slice_of_energy = pd.DataFrame()
# directory = os.getcwd()

# for dir in os.listdir(directory):
#     if os.path.isdir(dir) == True:
#         frequencies.append(int(dir))
# frequencies = sorted(list(set(frequencies)))
# ticks = [str(x) for x in frequencies]
# ticks.insert(0, "0")
# print('Frequencies is ', frequencies)  
# print('Frequencies is ', ticks)  

# for address, dirs, names in os.walk(directory):
#     for name in names:
#         filename, file_extension = os.path.splitext(name)
#         if file_extension == ".dat":
#             amino_acid = re.search(r'^\w{,2}[^\_]', filename)
#             amino_acid = amino_acid.group(0)
#             amino_acids.append(amino_acid)
# amino_acids = sorted(list(set(amino_acids)))
# print('Amino acids is ', amino_acids)

# for i in amino_acids:
#     for j in energy_type:
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')
#         out = pd.DataFrame()
#         for a in range(len(frequencies)):
#             f = frequencies[a]
#             data = pd.DataFrame()
#             files = os.listdir(directory+"/"+str(f)+'/'+i)
#             files = sorted(files)
#             for file in files:
#                 filename, file_extension = os.path.splitext(file)
#                 if file_extension == ".dat":
#                     one_file_data = pd.read_csv(
#                         directory+"/"+str(f)+'/'+i+"/"+file, delimiter=' ', index_col=None, header=[0])
#                     data[(str(filename)+"_"+j)] = round((one_file_data[j])*0.0434/5400, 3) # переводим усл.ед. в эВ
#             last_moment_energies = list()
#             for energy_column in data.columns.values:
#                 last_moment_energies.append(
#                     float(data.iloc[-1, data.columns.get_loc(energy_column)]))           
#             out[i+str(f)+j] = last_moment_energies
#         # print(out)
#             print('Processing of ', i+' '+str(f)+' '+j) 
            
#             X_Y_Spline = make_interp_spline(field_amplitudes, last_moment_energies)
#             X_ = np.linspace((np.array(field_amplitudes)).min(), (np.array(field_amplitudes)).max(), 500)
#             Y_ = X_Y_Spline(X_)
#             # There is third arg used for equal distanse in yTicks
#             ax.plot(X_, Y_, norm_freq[a], zdir='y', marker = 'o', markersize=0.3,  linewidth = 2)
#         ax.set_title((str(i)).upper()+' '+str(j))
#         ax.set_xlabel('Field Amplitude (V/nm)')
#         ax.set_ylabel('Field Frequency ($cm^{-1}$)')
#         ax.set_zlabel('Amplitude (eV)')
#         ax.set_yticklabels(ticks) # Для нормальной дистанции
#         ax.grid()
#         plt.legend([str(x)+'$cm^{-1}$' for x in frequencies])
#         # plt.show()
#         fig.set_size_inches(10, 10)
#         ax.figure.savefig(directory+'/'+str(i)+'_'+str(j).lower()+".png", dpi=1200)

def make_spectres():
    zero_data = pd.read_csv(
        '/Users/max/Yandex.Disk.localized/NAMD/basic_ak_no_field/ala.dat', delimiter=' ', index_col=None)
    zero_data.rename(columns={'0.0': 'Frequency',
                     '0.0.1': 'Amplitude'}, inplace=True)
    zero_data.insert(2, 'Amp×104', zero_data['Amplitude']*(10**4))

    folder = os.getcwd()
    files = os.listdir(folder)
    # file = open("frequencies.txt", "w")
    file = open(os.path.basename(folder) + "_ratio.txt", "w")
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

            max_amp_no_field = zero_data.loc[zero_data['Amplitude'].where((zero_data['Frequency'] < (float(
                max_amp_freq) + 1)) & (zero_data['Frequency'] > (float(max_amp_freq) - 1))).idxmax(), 'Amp×104']
            max_amp_freq_no_field = zero_data.loc[zero_data['Amplitude'].where((zero_data['Frequency'] < (float(
                max_amp_freq) + 1)) & (zero_data['Frequency'] > (float(max_amp_freq) - 1))).idxmax(), 'Frequency']

            # Ищем точку максимума без учета частоты поля
            # max_amp_freq = df.loc[df['Amplitude'].idxmax(), 'Frequency']
            # max_amp = df.loc[df['Amplitude'].idxmax(), 'Amp×104']

            # mess = 'Field ' + \
            #     str(field_freq) + " : " + str(max_amp_freq) + \
            #     ' - ' + str(max_amp) + '\n'

            ratio = max_amp / max_amp_no_field
            ratio = round(ratio, 2)
            mess = str(field_freq) + " : " + str(ratio) + '\n'
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

make_spectres()