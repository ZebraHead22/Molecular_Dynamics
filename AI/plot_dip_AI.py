import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def create_title(filename):
    # Extract the number and other components from the filename
    match = re.search(r'(\w+)_(\d+)_([a-zA-Z]+)', filename)
    if match:
        prefix = match.group(1).upper()
        number = match.group(2)
        environment = match.group(3).lower()

        if 'water' in environment:
            return f"{prefix} WATER N={number}"
        elif 'vac' in environment or 'vacuum' in environment:
            return f"{prefix} VACUUM N={number}"
        elif 'linear' in environment:
            return f"{prefix} LINEAR N={number}"
        elif 'cyclic' in environment:
            return f"{prefix} CYCLIC N={number}"
    return filename

def process_file(name):
    filename, file_extension = os.path.splitext(name)
    if file_extension == ".dat":
        title = create_title(filename)
        print(f"-- Generated title: {title}")

        # Read the .dat file into a DataFrame
        try:
            df = pd.read_csv(os.path.join(os.getcwd(), name), sep=' ')
        except Exception as e:
            print(f"Error reading {name}: {e}")
            return

        df.dropna(how='all', axis=1, inplace=True)
        df.rename(columns={'#': 'frame', 'Unnamed: 2': 'dip_x', 'Unnamed: 4': 'dip_y',
                           'Unnamed: 6': 'dip_z', 'Unnamed: 8': '|dip|'}, inplace=True)
        df.insert(1, "Time", (df['frame'] * 2 / 1000000))

        plt.gcf().clear()
        plt.plot(np.array(df["Time"].tolist()), np.array(df["|dip|"].tolist()), linewidth=1, color='black')
        plt.ylabel('Dipole Moment (D)')  
        plt.xlabel('Time (ns)')
        plt.title(title)
        plt.grid()
        
        # Save plots
        plt.savefig(f"{filename}_dip.eps", format='eps')
        plt.savefig(f"{filename}_dip.png", dpi=600)

def main():
    # Get the current working directory and list all files
    files = [name for _, _, names in os.walk(os.getcwd()) for name in names if name.endswith('.dat')]
    
    # Use 16 processes for multiprocessing
    num_processes = min(16, cpu_count())
    with Pool(processes=num_processes) as pool:
        pool.map(process_file, files)

if __name__ == '__main__':
    main()
