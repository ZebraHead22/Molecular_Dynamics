import os
import re
import matplotlib
import numpy as np
import plotly.express as px
import pandas as pd
import pylab as pl
from math import sqrt
from matplotlib import pyplot as plt
import plotly.express as px

# for file in os.listdir(os.getcwd()):
    
#     df = pd.read_csv(file, sep=" ", index_col=None)
#     df.rename(columns={'0.0': 'Frequency',  '0.0.1': 'Amplitude'}, inplace=True)
#     df["Frequency"] = [round(x) for x in df["Frequency"].tolist()]
#     df = df.set_index("Frequency")
#     df = df.groupby(level='Frequency').mean()
#     df = df.reset_index()
    
#     print(str(len(df["Frequency"]))+" "+file)

file = "/Volumes/DECAY_DATA/namd/spectres_high_frequency_fields/gly/gly_3333.dat"
df = pd.read_csv(file, sep=" ", index_col=None)
df.rename(columns={'0.0': 'Frequency',  '0.0.1': 'Amplitude'}, inplace=True)
df["Frequency"] = [round(x) for x in df["Frequency"].tolist()]
df = df.set_index("Frequency")
df = df.groupby(level='Frequency').mean()
max_value = df.loc[df["Amplitude"].idxmax(), "Amplitude"]
df[df["Amplitude"] < max_value*0.02] = np.nan
df = df.reset_index()
plt.plot(df["Frequency"], df["Amplitude"], 'o-')
plt.xlim((0, 6000))
plt.show()

# plt.scatter(np.array(df["Frequency"].tolist()), np.array(df["Amplitude"]).tolist(), cmap="viridis")
# plt.grid()
# plt.show()


