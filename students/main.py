import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re

list_template = list()
# глобальная цель - 6 датафреймов с Кин, Пот энергиями по трем АК
# 1 прочитать в цикле каждый ДФ
# 2 с помощью циклов получить список аминокислот

amino_acids = ['ff', 'trp', 'gly']

os.path.join
os.path.splitext
os.path.basename

#список ак - не знаю как сделать так, чтобы не выводилась "data"
dir = os.getcwd()
for address, dirs, files in os.walk(dir):
    for dir in dirs:
        name  = os.path.splitext(dir)
        list_template = list(name) 
        removed = list_template.pop(1)
        
        print(list_template) 



#чтение файлов  - не работает 
for address, dirs, names in os.walk(dir):
    for name in names:
        df = pd.read_csv(name)
        print(df.head())

