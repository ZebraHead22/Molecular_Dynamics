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

for file in os.listdir(os.getcwd()):
    
    df = pd.read_csv(file, sep=" ", index_col=None)
    df.rename(columns={'0.0': 'Frequency',  '0.0.1': 'Amplitude'}, inplace=True)
    df["Frequency"] = [round(x) for x in df["Frequency"].tolist()]
    df = df.drop_duplicates(subset=["Frequency"])
    print(str(len(df["Frequency"]))+" "+file)