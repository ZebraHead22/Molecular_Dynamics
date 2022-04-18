import os
import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack
from pandas import Series
from matplotlib import pyplot as plt

path="/Users/max/Yandex.Disk.localized/namd"
os.chdir(path)
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df1 = pd.read_csv("output1.csv")
df2 = pd.read_csv("output2.csv")
time = []
i=0
for i in range (0, 1500005, 5):
    time.append(i)
    i+=1

energyWithoutfield = df1["TOTAL3"].tolist()
energyWithFieldRaw = df2["TOTAL3"].tolist()
energyWithField = []
energyWithField.append(energyWithFieldRaw[0])

for i in range(1, (len(energyWithFieldRaw)-1),2):
     energyWithField.append((energyWithFieldRaw[i]+energyWithFieldRaw[i+1])/2)
    


# print(len(time))
print(energyWithField[2])