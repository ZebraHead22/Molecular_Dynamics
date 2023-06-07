import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dat_files = list()
data = pd.DataFrame()

directory = os.getcwd()

for address, dirs, names in os.walk(directory):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            dat_files.append(name)
            







