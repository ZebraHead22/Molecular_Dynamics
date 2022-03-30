import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

jls_extract_var = "/Users/max/Documents/namd/eField/FF_400/" #здесь открывать папку кнопкой
os.chdir(jls_extract_var)
folders = os.listdir(jls_extract_var)
ds_folder = '.DS_Store'
for i in folders:
    if i != ds_folder:
        df = pd.read_csv(jls_extract_var + i + '/' +'bond.csv')
        time = df['TS']
        energy = df['ENERGY']
        plt.plot(time, energy, 'r')
        plt.savefig(jls_extract_var + i + '/' +'fig')
    else:
        print("Zalupka")
