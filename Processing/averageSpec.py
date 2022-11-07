import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

folders = os.listdir(os.getcwd())
for folder in folders:
    if os.path.isdir(folder) == True:       
        #Define variable
        av = pd.DataFrame()
        dirPath = os.getcwd()+'/'+folder
        files = os.listdir(dirPath)
        print(dirPath)
        #Write data into one file
        for i in files:
            filename, file_extension = os.path.splitext(dirPath+'/'+i)
            if file_extension == ".dat":
                print(i)
                df = pd.read_csv(dirPath+'/'+i, delimiter=' ', index_col=None)
                df.rename(columns = {'0.0' : 'Freq', '0.0.1' : 'Amplitude'}, inplace = True)
                av['Freq'] = df['Freq']
                av[os.path.basename(i)] = df['Amplitude']
                #Make column with average amplitude
                av_average = av.drop('Freq', axis = 1)
                av['average'] = av_average.mean(axis = 1)
                with pd.ExcelWriter(str(dirPath)+"/"+'result.xlsx') as writer:
                    av.to_excel(writer, sheet_name='average data', index=None, index_label=None)
        #Make average amplitude data
        av_average = av.drop('Freq', axis = 1)
        av_average = np.array(av_average.mean(axis = 1))
        av_freq = np.array(av['Freq'])
        
        #Make plot
        plt.gcf().clear()
        plt.plot(av_freq, av_average)
        plt.ylabel('Spectral Density (a.u.)')
        plt.xlabel('Frequency ($cm^{-1}$)')
        plt.grid()
        plt.savefig(dirPath+'/'+os.path.basename(dirPath)+'_average.png')
        print('Picture saved...')
    else:
        pass
        continue

#Go to make global result
amp = []
time = []
time_raw = []
for folder in folders:
    if os.path.isdir(folder) == True: 
        dirPath = os.getcwd()+'/'+folder
        res_files = os.getcwd()+'/'+folder + '/' +'result.xlsx'
        xl_file = pd.ExcelFile(res_files)
        res = pd.read_excel(res_files, index_col=None)
        max_x = res['average'].where(res['Freq']>3000).max()
        amp.append(max_x)
        time_raw.append(re.findall(r'\d+', os.path.basename(dirPath)))

for i in time_raw:
    time.append(float(i[0]))

#Make dataframe
res_data = pd.DataFrame()
res_data['Time'] = time
res_data['Spectral Density'] = amp

#Write to xlsx
with pd.ExcelWriter(str(os.getcwd())+"/"+'result_data.xlsx') as writer:
    res_data.to_excel(writer, sheet_name='result data', index=None, index_label=None)

print('All done')



