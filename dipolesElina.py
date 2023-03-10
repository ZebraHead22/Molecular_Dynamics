import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import openpyxl

destination_path = "."
dip_path = "dip"


def print_dip(filename):
    base = os.path.basename(filename)
    general_name = os.path.splitext(base)[0]
    print(base)

    df = pd.read_excel(os.path.join(dip_path,filename))
    df=df[df.select_dtypes(include=[np.number]).ge(0).all(1)]
    # df = pd.read_csv(os.path.join(dip_path,filename), sep="  ")
    df.to_excel('result2.xlsx', sheet_name='average data', index=None, index_label=None)
   



dip_filenames = next(os.walk(dip_path), (None, None, []))[2]
for dip in dip_filenames:
    print_dip(dip)