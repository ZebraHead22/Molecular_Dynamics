import os
import pandas as pd
import matplotlib.pyplot as plt

# Step 1. Load data from CSV file.
folder = os.getcwd()
df = pd.read_csv(os.path.join(folder, 'data_peaks_summary.csv'))

DISTANCE = 150

# Step 2. Extract all frequencies, splitting cells with multiple values (separated by ';')
frequencies = []
for cell in df.values.flatten():
    if pd.isna(cell):
        continue
    # If the cell is a string and contains ';', split it
    if isinstance(cell, str) and ';' in cell:
        parts = cell.split(';')
        for part in parts:
            try:
                value = float(part)
                if value < 300:  # Skip frequencies less than 300
                    continue
                frequencies.append(value)
            except ValueError:
                pass
    else:
        try:
            value = float(cell)
            if value < 300:  # Skip frequencies less than 300
                continue
            frequencies.append(value)
        except ValueError:
            pass

frequencies = sorted(frequencies)

# Step 3. Group frequencies: add frequencies to a cluster if they are within 100 of the cluster's minimum frequency.
clusters = []
current_cluster = []

for freq in frequencies:
    if not current_cluster:
        current_cluster.append(freq)
    else:
        # Check difference with the first (minimum) frequency in the current cluster.
        if freq - current_cluster[0] <= int(DISTANCE):
            current_cluster.append(freq)
        else:
            clusters.append(current_cluster)
            current_cluster = [freq]
if current_cluster:
    clusters.append(current_cluster)

# Step 4. For each cluster, get the minimum value, maximum value, and the count of elements.
min_vals = [min(cluster) for cluster in clusters]
max_vals = [max(cluster) for cluster in clusters]
counts = [len(cluster) for cluster in clusters]

# Step 5. Plot the bar chart.
plt.figure(figsize=(12, 6))
for cluster in clusters:
    x = min(cluster)                   # Bar starts at the minimum frequency of the cluster
    width = max(cluster) - min(cluster)  # Bar width is the frequency range of the cluster
    count = len(cluster)
    plt.bar(x, count, width=width, align='edge', alpha=0.6, edgecolor='black')
    # Annotate with frequency range (min - max)
    plt.text(x, count, f'{min(cluster)} - {max(cluster)}', va='bottom', ha='left', fontsize=8)

plt.xlim(0, 4000)
plt.xlabel('Frequency Range')
plt.ylabel('Count of Frequencies')
plt.title(f'Grouping Similar Frequencies (within ±{int(DISTANCE)}) Across Experiments')
# plt.show()
plt.savefig("bar_plot.png", dpi=300)
