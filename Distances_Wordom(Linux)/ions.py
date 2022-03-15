import os
import xlsxwriter
import re
import itertools

# CONSTANTS
main_path = "D:/namd/salt2/trp_4/"
ionFile = "ion_data.xlsx"
pdbFile = 'ionized.pdb'

# CHANGE DIRECTORY
os.chdir(main_path)
concentrations =[]
ionsRes = []
res = []
def Molecules():
    global cz, expenses
    for dirpath, dirnames, filenames in os.walk("."):
        for dirname in dirnames:
            concentrations.append(float(dirname))
            os.chdir(main_path + dirname)
            with open(pdbFile, "r") as file:
                # итерация по строкам
                r=0
                for line in file:
                    if re.findall(r'ION', line):
                        r+=1
                        res.append(r)
    ions = []
    for i in range(len(res)-1):
        if res[i] > res[i+1]:
            if re.findall(r'_4', main_path):
                ions.append(round(res[i]/4))
            else:
                ions.append(res[i])
    if re.findall(r'_4', main_path):
        ions.append(round(res[-1]/4))
    else:
        ions.append(res[-1])
    concentrations.sort()
    if len(ions) == 14:
        ions.sort()
        expenses = [[concentrations[i], ions[i]] for i in range(len(concentrations))]
    if len(ions) == 13:
        ions.append(0)
        ions.sort()
        expenses = [[concentrations[i], ions[i]] for i in range(len(concentrations))]

    # WRITE IONS NUMBER PER CONCENTRATIONS IN ION_DATA FILE
    workbook = xlsxwriter.Workbook(main_path + ionFile)
    worksheet = workbook.add_worksheet('IONS')
    for i, (n, ion) in enumerate(expenses, start=1):
        worksheet.write(f'A{i}', n)
        worksheet.write(f'B{i}', ion)
    workbook.close()
#
Molecules()