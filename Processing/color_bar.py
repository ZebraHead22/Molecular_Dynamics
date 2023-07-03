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
        frequency = re.search(r'\d+', file)
        frequency = frequency.group(0)
        field_frequency.append(frequency)
        df = pd.read_csv(directory+'/'+file, delimiter=' ', index_col=None)
        df.rename(columns={'0.0': 'Frequency',
                      '0.0.1': 'Amplitude'}, inplace=True)
        data.merge(df, how='inner', left_index=True, right_index=True)

    print(data.head())


    #     for i in df['Frequency'].tolist():
    #         frequencies.append(i)
    # frequencies = list(set(frequencies))
    # frequencies.sort()
    # data['Frequency'] = frequencies
    # print(data.head())
    # for file in dat_files:
    #     df = pd.read_csv(directory+'/'+file, delimiter=' ', index_col=None)
    #     df.rename(columns={'0.0': 'Frequency',
    #                   '0.0.1': 'Amplitude'}, inplace=True)
    #     for j in frequencies:
    #         data[file] = df['Amplitude'].where(df['Frequency'].isin(j)==[j])
    # data = data.dropna()
    # print(data.head())






    #     amplitudes = df['Amplitude'].tolist()
    #     amplitudes = [round(float(x*(10**3))) for x in amplitudes]
    #     all_amplitudes.append(amplitudes)
    # # print(all_amplitudes)
    # fig = px.imshow(all_amplitudes,
    #                 labels=dict(x="$Frequency, cm^{-1}$", y="$FieldFrequency, cm^{-1}$", color="Spectral density, a.u."),
    #                 x=df["Frequency"].tolist(),
    #                 y=field_frequency)
    # fig.update_xaxes(side="top")
    # fig.write_image(file='./staff_plot.png', format='png')
        