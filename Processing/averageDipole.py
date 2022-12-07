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
av['average_dip_x'] = av_average[['1st dip_x', '2nd dip_x', '3rd dip_x']].mean(axis = 1)
av['average_dip_y'] = av_average[['1st dip_y', '2nd dip_y', '3rd dip_y']].mean(axis = 1)
av['average_dip_z'] = av_average[['1st dip_z', '2nd dip_z', '3rd dip_z']].mean(axis = 1)
av['average_dip_abs'] = av_average[['1st dip_abs', '2nd dip_abs', '3rd dip_abs']].mean(axis = 1)


with pd.ExcelWriter(str(os.getcwd())+"/"+'main_result.xlsx') as writer:
    av.to_excel(writer, sheet_name='Average data', index=None, index_label=None)


av_frame = np.array(av['Frames'].tolist())
av_x = np.array(av['average_dip_x'].tolist())
av_y = np.array(av['average_dip_y'].tolist())
av_z = np.array(av['average_dip_z'].tolist())
av_abs = np.array(av['average_dip_abs'].tolist())

# Make plot
plt.gcf().clear()
plt.scatter(av_frame, av_abs, c='deeppink', s=50, edgecolor='black')
plt.ylabel('Dipole moment (D)')
plt.xlabel('Frames (pci)')
plt.grid()
plt.savefig((os.getcwd())+"/"+'main_fig.png')
print('Picture saved...')

print('All done')

# plt.subplots(2,2)
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(av_frame, av_x)
# axs[0, 0].set_title('Dipole Moment Ox')
# axs[0, 1].plot(av_frame, av_y)
# axs[0, 1].set_title('Dipole Moment Oy')
# axs[1, 0].plot(av_frame, av_z)
# axs[1, 0].set_title('Dipole Moment Oz')
# axs[1, 1].plot(av_frame, av_abs)
# axs[1, 1].set_title('Dipole Moment |ABS|')

# for ax in axs.flat:
#     ax.set(xlabel='frame (u.)', ylabel='Dipole Moment (D)')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

# plt.savefig((os.getcwd())+"/"+'main_4g.png')