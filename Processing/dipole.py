# Здесь считаем полный дипольный момент для одной реализации для вычисления добротности


import os
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# df = pd.DataFrame()
files = os.listdir(os.getcwd())
dirPath = os.getcwd()
# dip_x_1 = []
# dip_y_1 = []
# dip_z_1 = []
# dip_abs_1 = []
# frames = []
for i in files:
    
    df = pd.DataFrame()
    files = os.listdir(os.getcwd())
    dirPath = os.getcwd()
    dip_x_1 = []
    dip_y_1 = []
    dip_z_1 = []
    dip_abs_1 = []
    frames = []
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        name = os.path.basename(i)
        print(i)
        df = pd.read_csv(dirPath+'/'+i, sep=' ')
        df.dropna(how='all', axis=1, inplace=True)
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                  'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
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
                            columns=['Time (ps)', 'dip_x', 'dip_y', 'dip_z', '|dip|'])


        # with pd.ExcelWriter(dirPath+'/'+'dependence.xlsx') as writer:
        #     dipole.to_excel(writer, sheet_name='Dipole', index=None, index_label=None)

        # dipole = dipole.loc[dipole['Time (ps)'] <= 40]

        # Make plot
        plt.gcf().clear()
        plt.plot(np.array(dipole["Time (ps)"].tolist()), np.array(dipole["dip_x"].tolist()), color='crimson', linewidth=2)
        plt.ylabel('Dipole Moment (D)')
        plt.xlabel('Time (ps)')
        plt.grid()
        plt.savefig(dirPath+"/"+name+'_dipoleMoment_X.png')
        print('Dip X picture saved...')

        plt.gcf().clear()
        plt.plot(np.array(dipole["Time (ps)"].tolist()), np.array(dipole["dip_y"].tolist()), color='darkmagenta', linewidth=2)
        plt.ylabel('Dipole Moment (D)')
        plt.xlabel('Time (ps)')
        plt.grid()
        plt.savefig(dirPath+"/"+name+'_dipoleMoment_Y.png')
        print('Dip Y picture saved...')

        plt.gcf().clear()
        plt.plot(np.array(dipole["Time (ps)"].tolist()), np.array(dipole["dip_z"].tolist()), color='indigo', linewidth=2)
        plt.ylabel('Dipole Moment (D)')
        plt.xlabel('Time (ps)')
        plt.grid()
        plt.savefig(dirPath+"/"+name+'_dipoleMoment_Z.png')
        print('Dip Z picture saved...')

        plt.gcf().clear()
        plt.plot(np.array(dipole["Time (ps)"].tolist()), np.array(dipole["|dip|"].tolist()), color='darkblue', linewidth=2)
        plt.ylabel('Dipole Moment (D)')
        plt.xlabel('Time (ps)')
        plt.grid()
        plt.savefig(dirPath+"/"+name+'_dipoleMoment_ABS.png')
        print('Dip ABS picture saved...')


        plt.gcf().clear()
        plt.plot(np.array(dipole["Time (ps)"].tolist()), np.array(dipole["dip_x"].tolist()), color='crimson', linewidth=1)
        plt.plot(np.array(dipole["Time (ps)"].tolist()), np.array(dipole["dip_y"].tolist()), color='darkmagenta', linewidth=1)
        plt.plot(np.array(dipole["Time (ps)"].tolist()), np.array(dipole["dip_z"].tolist()), color='green', linewidth=1)
        plt.plot(np.array(dipole["Time (ps)"].tolist()), np.array(dipole["|dip|"].tolist()), color='darkblue', linewidth=1)
        plt.ylabel('Dipole Moment (D)')
        plt.xlabel('Time (ps)')
        plt.grid()
        plt.legend(['dip_x', 'dip_y', 'dip_z', '|dip|' ])
        plt.savefig(dirPath+"/"+name+'_dipoleMoment_all.png')
        print('Dip ALL picture saved...')

# plt.gcf().clear()
# plt.plot(np.array(frames), np.array(dip_abs))
# plt.ylabel('Time (ps))')
# plt.xlabel('Frequency ($cm^{-1}$)')
# plt.grid()
# plt.show()
