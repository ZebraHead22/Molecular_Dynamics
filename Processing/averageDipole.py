import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



df = pd.DataFrame()
files = os.listdir(os.getcwd())

folders = os.listdir(os.getcwd())
for folder in folders:
    if os.path.isdir(folder) == True:       
        #Define variable
        av = pd.DataFrame()
        dirPath = os.getcwd()+'/'+folder
        files = os.listdir(dirPath)
        print(dirPath)
        frames = []
        dip_x_1 = []
        dip_y_1 = []
        dip_z_1 = []
        dip_abs_1 = []
        for i in files:
            filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
            if file_extension == ".dat":
                df = pd.read_csv(dirPath+'/'+i, sep = ' ')
                df.dropna(how='all', axis=1, inplace=True)
                df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y', 'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
                dip_x_1.append(df['dip_x'].tolist())
                dip_y_1.append(df['dip_y'].tolist())
                dip_z_1.append(df['dip_z'].tolist())
                dip_abs_1.append(df['|dip|'].tolist())

        dip_x = []
        dip_y = []
        dip_z = []
        dip_abs = []

        for i in dip_abs_1:
            for j in i:
                dip_abs.append(j)

        for i in dip_x_1:
            for j in i:
                dip_x.append(j)

        for i in dip_y_1:
            for j in i:
                dip_y.append(j)

        for i in dip_z_1:
            for j in i:
                dip_z.append(j)

        for i in range(len(dip_x)):
            frames.append(int(i)/200)

        dipole = pd.DataFrame(list(zip(frames, dip_x, dip_y, dip_z, dip_abs)),
                    columns =['Frames', 'dip_x', 'dip_y', 'dip_z', '|dip|'])

        
        with pd.ExcelWriter(dirPath+'/'+'dependence.xlsx') as writer:                 
            dipole.to_excel(writer, sheet_name='Dipole', index=None, index_label=None)

av = pd.DataFrame()
av_p = pd.DataFrame()
for folder in folders:
    if os.path.isdir(folder) == True: 
        df = pd.read_excel(os.getcwd()+'/'+folder+'/'+'dependence.xlsx')
        av['Frames'] = df['Frames']
        av[folder+' '+'dip_x'] = df['dip_x']
        av[folder+' '+'dip_y'] = df['dip_y']
        av[folder+' '+'dip_z'] = df['dip_z']
        av[folder+' '+'dip_abs'] = df['|dip|']

print(av.head())

av_average = av.drop('Frames', axis = 1)
av['average_dip_x'] = av_average[['first_exp dip_x', 'sec_exp dip_x']].mean(axis = 1)
av['average_dip_y'] = av_average[['first_exp dip_y', 'sec_exp dip_y']].mean(axis = 1)
av['average_dip_z'] = av_average[['first_exp dip_z', 'sec_exp dip_z']].mean(axis = 1)
av['average_dip_abs'] = av_average[['first_exp dip_abs', 'sec_exp dip_abs']].mean(axis = 1)


with pd.ExcelWriter(str(os.getcwd())+"/"+'main_result.xlsx') as writer:
    av.to_excel(writer, sheet_name='Average data', index=None, index_label=None)


av_frame = np.array(av['Frames'].tolist())
av_x = np.array(av['average_dip_x'].tolist())
av_y = np.array(av['average_dip_y'].tolist())
av_z = np.array(av['average_dip_z'].tolist())
av_abs = np.array(av['average_dip_abs'].tolist())

# # Make plot
plt.gcf().clear()
plt.scatter(av_frame, av_abs, c='deeppink', s=100, edgecolor='black')
plt.ylabel('Dipole moment (D)')
plt.xlabel('Frames (pci)')
plt.grid()
plt.savefig((os.getcwd())+"/"+'main_fig.png')
print('Picture saved...')

print('All done')