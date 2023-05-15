import os
import re
import matplotlib
import numpy as np
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt

# Define constant lists
FREQUENCIES_GLY = list()
FREQUENCIES_FF = list()
FREQUENCIES_TRP = list()
# Function body
am_folder = os.getcwd()
for address, dirs, files in os.walk(am_folder):
    for name in files:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            df = pd.read_csv(
                os.path.join(address, name), delimiter=' ', index_col=None)
            df.rename(columns={'0.0': 'Frequency',
                                '0.0.1': 'Amplitude'}, inplace=True)
            frequency = re.search(
                r'\d+', str(os.path.basename(filename)))
            frequency = frequency.group(0)
            closest_value_min = df.iloc[(
                df['Frequency']-float(int(frequency)-20)).abs().argsort()[:1]].index.tolist()
            closest_value_max = df.iloc[(
                df['Frequency']-float(int(frequency)+20)).abs().argsort()[:1]].index.tolist()
            max_amplitude = df.loc[closest_value_min[0]: closest_value_max[0], 'Amplitude'].max()
            max_amplitude_frequency = df.loc[df['Amplitude']
                                                == max_amplitude, 'Frequency'].values[0]
            amino = re.search(r'^[\w]{1,2}[^\_]', str(
                os.path.basename(filename)))
            amino = amino.group(0)
            if amino == "gly":
                FREQUENCIES_GLY.append(max_amplitude_frequency)
            elif amino == "ff":
                FREQUENCIES_FF.append(max_amplitude_frequency)
            elif amino == "trp":
                FREQUENCIES_TRP.append(max_amplitude_frequency)
# Calculation frequencies in THz
FREQUENCIES_THZ_GLY = [float(x*0.03) for x in FREQUENCIES_GLY]
FREQUENCIES_THZ_FF = [float(x*0.03) for x in FREQUENCIES_FF]
FREQUENCIES_THZ_TRP = [float(x*0.03) for x in FREQUENCIES_TRP]
# Make plot
plt.gcf().clear()
# Make subplots
fig, ax = plt.subplots()

def cm_to_inch(value):  # Define picture size, calc cm in inch
    return value/2.54
fig.set_figheight(cm_to_inch(26))  # 26 cm in height
fig.set_figwidth(cm_to_inch(16))  # 16 cm in width
# Second plot for second y-axis
ax_e = ax.twinx()
ax.eventplot(FREQUENCIES_GLY, orientation="vertical",
                lineoffsets=-1.5, linewidth=0.75, color="black")
ax.eventplot(FREQUENCIES_TRP, orientation="vertical",
                lineoffsets=0, linewidth=0.75, color='black')
ax.eventplot(FREQUENCIES_FF, orientation="vertical",
                lineoffsets=1.5, linewidth=0.75, color="black")
# Set other graph parameters: labels, grid, limits;
# For collective fluctuations use 0-200 cm-1 (0-6 THz) range;
# For local fluctuations use 200-1000 cm-1 (6-30 THz)  range.
ax.set_ylabel('Frequency ($cm^{-1}$)')
ax_e.set_ylabel('Frequency (THz)')
ax.set_ylim(0, 1000)
ax_e.set_ylim(0, 30)
ax.text(-1.8, -80, 'Glycine                 Tryptophan        Diphenylalanine')
ax.set_xticks([])
figure_name = os.getcwd() + '/' + "evenplot_all.png"
fig.savefig(figure_name)
print("Figure save in " + str(os.path.dirname(figure_name)) +
        " as " + str(os.path.basename(figure_name)))
