import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from matplotlib import cm
import matplotlib.colors as mcolors

def plot_waterfall_3d(filepath):
    df = pd.read_excel(filepath)

    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    aa_name = base_name.split('_')[0]

    field_amplitudes = df.iloc[:, 1].values

    potential_columns = [col for col in df.columns if col.endswith("POTENTIAL") and col.startswith(aa_name)]
    freq_values = [int(col.replace(aa_name, '').replace("POTENTIAL", '')) for col in potential_columns]

    freq_and_cols = sorted(zip(freq_values, potential_columns))
    freq_values, sorted_columns = zip(*freq_and_cols)

    # Увеличиваем размер фигуры для лучшего размещения подписей
    fig = plt.figure(figsize=(14, 10))  # Было (13, 9)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.08, right=3, bottom=0.08, top=0.92)

    y_positions = range(len(sorted_columns))
    
    min_energy_global = min([df[col].min() for col in sorted_columns])
    
    num_lines = len(sorted_columns)
    colormap = cm.get_cmap('cool', num_lines)
    
    polygons = []
    colors_list = []
    
    x_smooth = np.linspace(field_amplitudes.min(), field_amplitudes.max(), 300)

    for y_index, (freq, col) in zip(y_positions, zip(freq_values, sorted_columns)):
        energy_values = df[col].values
        color = colormap(y_index / num_lines)
        
        cs = CubicSpline(field_amplitudes, energy_values)
        energy_smooth = cs(x_smooth)
        
        ax.plot(
            x_smooth, 
            [y_index] * len(x_smooth),
            energy_smooth, 
            linewidth=1.5,
            color=color
        )
        
        verts = []
        for x_val, z_val in zip(x_smooth, energy_smooth):
            verts.append((x_val, z_val))
        for x_val, z_val in zip(reversed(x_smooth), [min_energy_global] * len(x_smooth)):
            verts.append((x_val, z_val))
        verts.append((x_smooth[0], energy_smooth[0]))
        
        polygons.append(verts)
        colors_list.append(color)

    poly_collection = PolyCollection(
        polygons,
        facecolors=[mcolors.to_rgba(c, alpha=0.2) for c in colors_list],
        edgecolors='none'
    )
    ax.add_collection3d(poly_collection, zs=y_positions, zdir='y')

    # Увеличиваем отступы для осей
    ax.set_xlabel('Field Amplitude', fontsize=12, labelpad=15)  # Увеличен labelpad
    ax.set_ylabel('Frequency (cm⁻¹)', fontsize=12, labelpad=15)  # Увеличен labelpad
    # ax.zaxis.set_rotate_label(False)  # Отключить автоматический поворот
    ax.set_zlabel('Energy', fontsize=12, labelpad=0)  # Значительно увеличен labelpad для оси Z

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([str(freq) for freq in freq_values])
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    ax.view_init(elev=25, azim=-45)
    
    # Увеличиваем общий отступ для графика
    fig.tight_layout(pad=5.0, rect=[0.05, 0.05, 3, 5])  # Увеличен pad

    output_folder = os.path.join(os.path.dirname(filepath), "graphs")
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(filepath))[0] + "_waterfall.png"
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}")

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".xlsx") and not filename.startswith("._"):
            full_path = os.path.join(folder_path, filename)
            try:
                plot_waterfall_3d(full_path)
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

if __name__ == "__main__":
    folder = os.getcwd()
    process_folder(folder)