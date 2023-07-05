import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re

# глобальная цель - 6 датафреймов с Кин, Пот энергиями по трем АК
# 1 прочитать в цикле каждый ДФ
# 2 с помощью циклов получить список аминокислот
# os.path.join
# os.path.splitext
# os.path.basename

list_template = list()
amino_acids = list()

dir = os.getcwd()
for address, dirs, files in os.walk(dir):
    for i in dirs:
        amino_acids.append(i)

for acid in amino_acids:
    amino_dir = dir+"/"+acid
    for file in os.listdir(amino_dir):
        filename, file_extension = os.path.splitext(file)
        if file_extension == ".dat":
            df = pd.read_csv(amino_dir+"/"+file, sep=" ", index_col=None)
            #pd.insert( ... , 1)
            #df["TIME"] = ...
            #ts =  df["TS"].tolist()
            df["TIME"] = [... for x in ts]
            print(df.head())
# в каждом TS 1 фс 
# посчитать новые значения для TIME, удалить TS, вставить TIME