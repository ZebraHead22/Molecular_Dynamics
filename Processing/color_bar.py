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

amino_acids = list()


directory = os.getcwd()
for address, dirs, names in os.walk(directory):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            amino_acid = re.search(r'^\w{,2}[^\_]', filename)
            amino_acid = amino_acid.group(0)
            amino_acids.append(amino_acid)
amino_acids = list(set(amino_acids))

for i in amino_acids:
    directory = os.getcwd()
    directory = directory+"/"+i
    dat_files = list()
    field_frequency = list()
    frequencies = list()
    all_amplitudes = list()
    data = pd.DataFrame()
    for file in os.listdir(directory):
        dat_files.append(file)
    dat_files.sort()
    for file in dat_files:
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".dat":
            frequency = re.search(r'\d+', file)
            frequency = frequency.group(0)
            field_frequency.append(frequency)
            df = pd.read_csv(directory+'/'+file, delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                               '0.0.1': 'Amplitude'}, inplace=True)
            df["Frequency"] = [round(x) for x in df["Frequency"].tolist()]
            df = df.drop_duplicates(subset=["Frequency"])

            amplitudes = df['Amplitude'].tolist()
            amplitudes = [round(float(x*(10**9))) for x in amplitudes]
            all_amplitudes.append(amplitudes)
    fig = px.imshow(all_amplitudes,
                    labels=dict(
                        x="$Frequency, cm^{-1}$", y="$FieldFrequency, cm^{-1}$", color="Spectral density, a.u."),
                    x=df["Frequency"].tolist(),
                    y=field_frequency)
    fig.update_xaxes(side="top")
    fig.write_image(file="./"+i+'_staff_plot.png', format='png')
