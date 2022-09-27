import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

av = pd.DataFrame()
time = []
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
                time.append(int(os.path.basename(filename)))
                amplitude.append(float(df['Amplitude'].where(df['Frequency']>3000).max()))
        exp = pd.DataFrame(list(zip(time, amplitude)), columns =['Time (ps)', 'Spectral Density (a.u.)'])
        print(exp)
        with pd.ExcelWriter(dirPath+'/'+'dependence.xlsx') as writer:
            exp.to_excel(writer, sheet_name='Dependence', index=None, index_label=None)
        time = []
        amplitude = []

for folder in folders:
    if os.path.isdir(folder) == True: 
        df = pd.read_excel(os.getcwd()+'/'+folder+'/'+'dependence.xlsx')
        print(df.head())
        av['Time'] = df['Time (ps)']
        av[folder] = df['Spectral Density (a.u.)']

print(av.head())

av_average = av.drop('Time', axis = 1)
av['average'] = av_average.mean(axis = 1)

print(av.head())

with pd.ExcelWriter(str(os.getcwd())+"/"+'main_result.xlsx') as writer:
            av.to_excel(writer, sheet_name='Average data', index=None, index_label=None)


av_freq = np.array(av['Time'].tolist())
av_average = np.array(av['average'].tolist())
# Make plot
plt.gcf().clear()
plt.plot(av_freq, av_average)
plt.ylabel('Spectral Density (a.u.)')
plt.xlabel('Time (ps)')
plt.grid()
plt.savefig((os.getcwd())+"/"+'main_fig.png')
print('Picture saved...')

print('All done')


#                 #Make column with average amplitude
#                 av_average = av.drop('Freq', axis = 1)
#                 av['average'] = av_average.mean(axis = 1)
#                 with pd.ExcelWriter(str(dirPath)+"/"+'result.xlsx') as writer:
#                     av.to_excel(writer, sheet_name='average data', index=None, index_label=None)
#         #Make average amplitude data
#         av_average = av.drop('Freq', axis = 1)
#         av_average = np.array(av_average.mean(axis = 1))
#         av_freq = np.array(av['Freq'])
#         #
#     else:
#         pass
#         continue

# #Go to make global result
# amp = []
# time = []
# time_raw = []
# for folder in folders:
#     if os.path.isdir(folder) == True: 
#         dirPath = os.getcwd()+'/'+folder
#         res_files = os.getcwd()+'/'+folder + '/' +'result.xlsx'
#         xl_file = pd.ExcelFile(res_files)
#         res = pd.read_excel(res_files, index_col=None)
#         max_x = res['average'].where(res['Freq']>3000).max()
#         amp.append(max_x)
#         time_raw.append(re.findall(r'\d+', os.path.basename(dirPath)))

# for i in time_raw:
#     time.append(float(i[0]))

# #Make dataframe
# res_data = pd.DataFrame()
# res_data['Time'] = time
# res_data['Spectral Density'] = amp

# #Write to xlsx
# with pd.ExcelWriter(str(os.getcwd())+"/"+'result_data.xlsx') as writer:
#     res_data.to_excel(writer, sheet_name='result data', index=None, index_label=None)

# print('All done')



