import os
import xlsxwriter

# CONSTANTS
concentrations = []
atomsNum = []
main_path = "D:/namd/salt2/trp_4/"
logFile = "min.log"
datFile = "multiplot.dat"
rawFile = "raw_data.xlsx"
ionFile = "mol_data.xlsx"

# CHANGE DIRECTORY
os.chdir(main_path)

# -----------------------------------------------------------------------------------------------------------------------
# DEF CREATE ION_FILE
# GETTING NUMBER OF ATOMS AND FORMING DOUBLE ARRAY
def Molecules():
    for dirpath, dirnames, filenames in os.walk("."):
        for dirname in dirnames:
            concentrations.append(float(dirname))
            # TAKE ATOMS NUMBER
            os.chdir(main_path + dirname)
            with open(logFile, 'r') as f:
                for i in range(181):
                    x = f.readline()
            l = len(x)
            integ = []
            i = 0
            while i < l:
                s_int = ''
                a = x[i]
                while '0' <= a <= '9':
                    s_int += a
                    i += 1
                    if i < l:
                        a = x[i]
                    else:
                        break
                i += 1
                if s_int != '':
                    integ.append(int(s_int))
            atoms = int(integ[0])
            atomsNum.append(atoms)
    expenses = [[concentrations[i], atomsNum[i]] for i in range(len(concentrations))]
    expenses.sort(key=lambda x: x[0])

    # WRITE IONS NUMBER PER CONCENTRATIONS IN ION_DATA FILE
    workbook = xlsxwriter.Workbook(main_path + ionFile)
    worksheet = workbook.add_worksheet('IONS')
    for i, (conc, ion) in enumerate(expenses, start=1):
        worksheet.write(f'A{i}', conc)
        worksheet.write(f'B{i}', ion)
    workbook.close()

Molecules()