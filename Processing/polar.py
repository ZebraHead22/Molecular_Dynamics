# Здесь считаем дипольные моменты в полярных координатах. строим годограф
import os
import re
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
for i in os.listdir(os.getcwd()):
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        df = pd.read_csv(os.getcwd()+'/'+i, sep=' ')
        df.dropna(how='all', axis=1, inplace=True)
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                  'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
        df.insert(5, 'r', (df['dip_x']**2+df['dip_y']**2)**(1/2))
        theta = list()
        for i, row in df.iterrows():
            if df.loc[i, 'dip_x'] > 0 and df.loc[i, 'dip_y'] >= 0:
                theta.append(
                    float(np.arctan(df.loc[i, 'dip_y']/df.loc[i, 'dip_x'])))
            elif df.loc[i, 'dip_x'] > 0 and df.loc[i, 'dip_y'] < 0:
                theta.append(
                    float(np.arctan(df.loc[i, 'dip_y']/df.loc[i, 'dip_x'])) + 2*np.pi)
            elif df.loc[i, 'dip_x'] < 0:
                theta.append(
                    float(np.arctan(df.loc[i, 'dip_y']/df.loc[i, 'dip_x'])) + np.pi)
            elif df.loc[i, 'dip_x'] == 0 and df.loc[i, 'dip_y'] > 0:
                theta.append(float(np.pi/2))
            elif df.loc[i, 'dip_x'] == 0 and df.loc[i, 'dip_y'] < 0:
                theta.append(float(3*np.pi/2))
            if df.loc[i, 'dip_x'] == 0 and df.loc[i, 'dip_y'] == 0:
                theta.append(None)
        df.insert(6, 'theta', theta)
        del_data = df[df['frame'] > 50]
        df = df.drop(del_data.index, axis=0)
        print(df)
        plt.gcf().clear()
        # plt.polar(np.array(df['theta'].tolist()), np.array(df['r'].tolist()), c='red')
        # plt.grid(True)
        # plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        p = ax.scatter(np.array(df['theta'].tolist()), np.array(df['r'].tolist()), color = 'red', s=25)
        # plt.show()
        plt.savefig(filename+'.png')