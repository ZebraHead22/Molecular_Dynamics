# Program for Wordom
import os
from matplotlib import pyplot as plt
import csv
import pandas
QT_LOGGING_RULES="qt5ct.debug=false"
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("Current Working Directory: ", os.getcwd())
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# New empty lists
numbers_per_line = []
min_distance_line = []
min_distance_col = []
in_one_line = []
distances = []
molecules = []
frames = []
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create .inp file
file = open('distance.txt', 'w')
file.write('BEGIN distance' + '\n')
lastSegNumA = input('Номер последнего сегмента AР: ')
lastSegNumB = input('Теперь ВР: ')
print('Последние сегменты АР и ВР ' + str(lastSegNumA) + '/' + str(lastSegNumB))
for i in range(1, int(lastSegNumA)+1):
    text = ('--TITLE ' + str(i) + '\n' + '--SELE /' + 'AP' + str(i) + '/*/* : /AUM/*/*' + '\n')
    file.write(text)
for i in range(1, int(lastSegNumB)+1):
    text2 = ('--TITLE ' + str(i+int(lastSegNumA)) + '\n' + '--SELE /' + 'BP' + str(i) + '/*/* : /AUM/*/*' + '\n')
    
    file.write(text2)
file.write('END' + '\n')
file.close()
print('Обработка')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Change extension
pre, ext = os.path.splitext('distance.txt')
os.rename('distance.txt', pre + '.inp')
print('Создан inp файл')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run wordom
print('Работа Wordom...')
os.system('wordom -iA distance.inp -imol solvate.pdb -itrj runned.dcd -otxt distances.txt')

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Read file
print('Читаю файл')
file = open('distances.txt', 'r')
lines = file.readlines()[1:-1]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Make one line from .txt
for line in lines:
    in_one_line.append(line)

print('Обрабатываю данные')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Make one line with numbers: frame and distance
for element in in_one_line:
    l = len(element)
    i = 0
    while i < l:
        space = ''
        symbol = element[i]
        while '0' <= symbol <= '9' or symbol == '.':
            space += symbol
            i += 1
            if i < l:
                symbol = element[i]
            else:
                break
        i += 1
        if space != '':
            min_distance_line.append(space)
min_distance_line.append('0')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Make 2 lists
for element in min_distance_line:
    if len(element) > 4:
        min_distance_col.append(float(element))
    else:
        distances.append(min_distance_col)
        min_distance_col = []
distances.pop(0)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create double list with necessary distances
dd = input('Каково минимальное расстояние?  ')
minDist = []
for frame in distances:
    n = [j for j in frame if j < int(dd)]
    minDist.append(n)
print('Подготовка к выводу данных')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create Y-axis for plot
for frame in minDist:
    molecules.append(len(frame))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create X-axis for plot
for i in range(len(in_one_line) + 1):
    frames.append(i)
frames.pop(0)
# print(len(frames))
# print(len(molecules))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Make plot
print('Рисую график')
plt.plot(frames, molecules)
plt.grid()
#  Add axis caption:
plt.xlabel('TS',
           fontsize=15,
           color='black')

plt.ylabel('molecules',
           fontsize=15,
           color='black')

plt.savefig('saved_plot.png', dpi=1000)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create xls file for graphics
print('Создаю таблицу значений')


df = pandas.DataFrame()

# Creating two columns
df['TS'] = frames
df['molecules'] = molecules

# Converting to excel
df.to_excel('result.xlsx', index=False)
print('Готово')