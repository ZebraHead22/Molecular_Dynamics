import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

name = os.getcwd()
basename = str(os.path.basename(name))
filename, file_extension = os.path.splitext(name)
files = os.listdir(name)


class Molecular_data:

    def __init__(self):
        self.df = pd.read_csv(self, sep=' ')
        self.df.dropna(how='all', axis=1, inplace=True)
        self.df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                  'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)  
              
    
    def print_head(self):
        print(self.df.head())

        
for i in files:
    filename, file_extension = os.path.splitext(i)
    if  file_extension == ".dat":
        print(i)
        # i = Molecular_data()

