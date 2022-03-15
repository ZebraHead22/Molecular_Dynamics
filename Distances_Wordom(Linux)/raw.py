import os
import xlsxwriter
import re
import pandas as pd
import openpyxl

# CONSTANTS
rawEnergies_1=[]
rawEnergies_2 =[]
rawEnergy=[]
even=[]
main_path = "/Users/max/Documents/my_projects/PolyPy/MD/test/"
logFile = "min.log"
datFile = "multiplot.dat"
rawFile = "raw_data.xlsx"

# CHANGE DIRECTORY
os.chdir(main_path)

# -----------------------------------------------------------------------------------------------------------------------
def Multi():
    fileZero = "/Users/max/Documents/my_projects/PolyPy/MD/test/1/multiplot.dat"
    fileZero = open(fileZero, 'r')
    timeLine = fileZero.readline()
    while True:
        timeValue = re.findall(r"\d+", timeLine)
        print(timeValue)

    # global rawEnergies_1
    # for dirpath, dirnames, filenames in os.walk("."):
    #     for dirname in dirnames:
    #         os.chdir(main_path + dirname)
    #         file = datFile
    #         file = open(file, 'r')
    #         count = 0
    #         while True:
    #             line = file.readline()
    #             energyRes = re.findall(r"-\d+.\d+", line)
    #             if energyRes != []:
    #                 rawEnergies_1.append(energyRes)
    #                 count+=1
    #                 z = []
    #                 if count == 1001:
    #                     for case in rawEnergies_1:
    #                         for i in case:
    #                             z.append(float(i))
    #                     rawEnergies_2.append(z)
    #                     rawEnergies_1 = []
    #
    #             if not line:
    #                 break
    #






Multi()