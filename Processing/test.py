import os
import re
path = os.getcwd()
files = os.listdir(os.getcwd())
for i in files:
    filename, file_extension = os.path.splitext(os.getcwd()+'/'+i)
    if file_extension == ".dat":
        print(i)