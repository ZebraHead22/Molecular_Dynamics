import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def one_experiment():
    av = pd.DataFrame()
    field_amplutide = []
    amplitude = []
    list_files = []

    folders = os.listdir(os.getcwd())
    for folder in folders:
        if os.path.isdir(folder) == True:     
            dirPath = os.getcwd()+'/'+folder
            files = os.listdir(dirPath)
            for i in files:
                filename, file_extension = os.path.splitext(dirPath+'/'+i)
                if file_extension == ".dat":
                    print(i)
                    df = pd.read_csv(dirPath+'/'+i, delimiter=' ', index_col=None)
                    df.rename(columns = {'0.0' : 'Frequency', '0.0.1' : 'Amplitude'}, inplace = True)
                    print(df.head())
                    dfFreq = np.array(df['Frequency'].tolist())
                    dfAmp = np.array(df['Amplitude'].tolist())
                    #make file xlsx
                    with pd.ExcelWriter(filename+'.xlsx') as writer:
                        df.to_excel(writer, sheet_name='Spectral Dencity', index=None, index_label=None)
                    #Make png
                    plt.gcf().clear()
                    plt.plot(dfFreq, dfAmp)
                    plt.ylabel('Spectral Density (a.u.)')
                    plt.xlabel('Frequency ($cm^{-1}$)')
                    plt.grid()
                    plt.savefig(filename+'.png')
                    print('Picture saved'+' '+filename+'.png')
                    #Find max over 3333cm-1 and add to list
                    field_amplutide.append(float(folder))
                    amplitude.append(df['Amplitude'].where(df['Frequency']>3000).max())
    exp = pd.DataFrame(list(zip(field_amplutide, amplitude)), columns =['Field Fmplutide (kcal/mol*A*e)', 'Spectral Density (a.u.)'])
    with pd.ExcelWriter('dependence.xlsx') as writer:
        exp.to_excel(writer, sheet_name='Dependence', index=None, index_label=None)

def few_experiments():
    av = pd.DataFrame()
    field_amplutide = []
    amplitude = []
    list_files = []

    folders = os.listdir(os.getcwd())
    for folder in folders:
        if os.path.isdir(folder) == True:       
            #Define variable
            dirPath = os.getcwd()+'/'+folder
            folders_2 = os.listdir(dirPath)
            for i in folders_2:
                exp_folder = dirPath+'/'+i
                files = os.listdir(exp_folder)
                for a in files:
                    # filename, file_extension = os.path.splitext(dirPath+'/'+a)
                    # if file_extension == ".dat":
                    #     print(a)
                    #     df = pd.read_csv(exp_folder+'/'+a, delimiter=' ', index_col=None)
                    #     df.rename(columns = {'0.0' : 'Frequency', '0.0.1' : 'Amplitude'}, inplace = True)
                    #     print(df.head())
                    #     dfFreq = np.array(df['Frequency'].tolist())
                    #     dfAmp = np.array(df['Amplitude'].tolist())
                    #     #make file xlsx
                    #     with pd.ExcelWriter(filename+'.xlsx') as writer:
                    #         df.to_excel(writer, sheet_name='Spectral Dencity', index=None, index_label=None)
                    #     #Make png
                    #     plt.gcf().clear()
                    #     plt.plot(dfFreq, dfAmp)
                    #     plt.ylabel('Spectral Density (a.u.)')
                    #     plt.xlabel('Frequency ($cm^{-1}$)')
                    #     plt.grid()
                    #     plt.savefig(filename+'.png')
                    #     print('Picture saved'+' '+filename+'.png')
                    #     #Find max over 3333cm-1 and add to list
                    #     field_amplutide.append(float(folder))
                    #     amplitude.append(df['Amplitude'].where(df['Frequency']>3000).max())
         


                




few_experiments()
# for folder in folders:
#     if os.path.isdir(folder) == True: 
#         df = pd.read_excel(os.getcwd()+'/'+folder+'/'+'dependence.xlsx')
#         print(df.head())
#         av['Time'] = df['Time (ps)']
#         av[folder] = df['Spectral Density (a.u.)']

# print(av.head())

# av_average = av.drop('Time', axis = 1)
# av['average'] = av_average.mean(axis = 1)

# print(av.head())

# with pd.ExcelWriter(str(os.getcwd())+"/"+'main_result.xlsx') as writer:
#             av.to_excel(writer, sheet_name='Average data', index=None, index_label=None)


# av_freq = np.array(av['Time'].tolist())
# av_average = np.array(av['average'].tolist())
# # Make plot
# plt.gcf().clear()
# plt.plot(av_freq, av_average)
# plt.ylabel('Spectral Density (a.u.)')
# plt.xlabel('Time (ps)')
# plt.grid()
# plt.savefig((os.getcwd())+"/"+'main_fig.png')
# print('Picture saved...')

# print('All done')






