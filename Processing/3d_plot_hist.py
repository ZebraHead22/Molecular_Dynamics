import os
import re
import numpy as np
import pandas as pd
import openpyxl

FIELD_FREQUENCIES = list()
DATA = pd.DataFrame()

for address, dirs, names in os.walk(os.getcwd()):
    for name in names:
        filename, file_extension = os.path.splitext(name)
        if file_extension == ".dat":
            field_frequency = re.search(r'\d+', filename)
            FIELD_FREQUENCIES.append(int(field_frequency.group(0)))

            df = pd.read_csv(os.getcwd()+'/'+name, sep=' ')
            df.dropna(how='all', axis=1, inplace=True)
            df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                            'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
            df.insert(0, "Time", (df['frame'] * 5)/1000)
            if filename == '0':
                DATA['No_Field'] = df['|dip|']
            else:
                DATA[filename] = df['|dip|']

FIELD_FREQUENCIES.sort()
FIELD_FREQUENCIES = [str(x) for x in FIELD_FREQUENCIES]
FIELD_FREQUENCIES[0] = 'No_Field'
DATA = DATA[FIELD_FREQUENCIES]
DATA.insert(0, 'Time_(ps)', DATA.index * 5 /1000) 
DATA = DATA.dropna(axis=0)
DATA.to_excel(os.getcwd()+'/5mol_data.xlsx', 'dip_data')