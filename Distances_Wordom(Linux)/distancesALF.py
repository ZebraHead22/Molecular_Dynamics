# Program for Wordom
import os
from matplotlib import pyplot as plt
import csv
import pandas
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
# path = input('Cnahge folder: ')
# os.chdir(path)
file = open('distance.txt', 'w')
file.write('BEGIN distance' + '\n')
alp = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
       'W', 'X', 'Y', 'Z']
segments = []
for i in alp:
    for j in alp:
        segments.append('PR' + i + j)
lastSeg = input('Please, write last segment: ')
message = str(lastSeg)
print('Last segment is ' + message.upper())
global id
id = segments.index(message.upper())
script = []

for number, seg in enumerate(segments):
    if number == id + 1:
        break
    else:
        text = ('--TITLE ' + str(number) + '\n' + '--SELE /' + seg + '/*/* : /AUM/*/*' + '\n')
        file.write(text)
file.write('END' + '\n')
file.close()
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Change extension
pre, ext = os.path.splitext('distance.txt')
os.rename('distance.txt', pre + '.inp')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run wordom
os.system('wordom -iA distance.inp -imol sol.pdb -itrj runned.dcd -otxt distances.txt')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Read file
file = open('distances.txt', 'r')
lines = file.readlines()[1:-1]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Make one line from .txt
for line in lines:
    in_one_line.append(line)
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
    if len(element) > 3:
        min_distance_col.append(float(element))
    else:
        distances.append(min_distance_col)
        min_distance_col = []
distances.pop(0)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create double list with necessary distances
minDist = []
for frame in distances:
    n = [j for j in frame if j < 55]
    minDist.append(n)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create Y-axis for plot
for frame in minDist:
    molecules.append(len(frame))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create X-axis for plot
for i in range(len(in_one_line) + 1):
    frames.append(i)
frames.pop(0)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Make plot
plt.plot(frames, molecules)
plt.grid()
#  Add axis caption:
plt.xlabel('t (ns)',
           fontsize=15,
           color='black')

plt.ylabel('molecules',
           fontsize=15,
           color='black')

plt.savefig('saved_plot.png', dpi=1000)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create xls file for graphics


df = pandas.DataFrame()

# Creating two columns
df['t, ns'] = frames
df['molecules'] = molecules

# Converting to excel
df.to_excel('result.xlsx', index=False)