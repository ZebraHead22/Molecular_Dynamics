import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv
import xlrd

file = "D:/localNamd/trp_Afrom_E/water/result.xlsx"

data_xls = pd.read_excel(file, 'first_exp', dtype=str, index_col=None)
data_xls.to_csv('D:/localNamd/trp_Afrom_E/water/csvfile.csv', encoding='utf-8', index=False)
df = pd.read_csv('D:/localNamd/trp_Afrom_E/water/csvfile.csv', sep=",", index_col=None, header=None)
fieldAmp = df.iloc[0].to_list()
fieldAmp.pop(0)
fieldAmp = np.array(fieldAmp)/10

amp = df.iloc[50001].to_list()
amp.pop(0)
amp = np.array(amp)

print(fieldAmp)
print(amp)