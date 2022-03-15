import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "/Users/max/Documents/my_projects/PolyPy/MD/test/aaa.xlsx"
extremums=[]
df = pd.read_excel(file)
excelData = df.to_numpy()
z = np.array(excelData)
rawData = z.transpose()
i=0
for element in rawData[1]:
    print(abs(float(element[i+1]-element[i])))
    i+=1

