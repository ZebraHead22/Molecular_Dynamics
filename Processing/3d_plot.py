import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline
from matplotlib import cm

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

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    y_positions = range(len(sorted_columns))
    polys = []

    # Количество линий
    num_lines = len(sorted_columns)
    # Используем colormap
    colormap = cm.get_cmap('cool', num_lines)

    for y_index, (freq, col) in zip(y_positions, zip(freq_values, sorted_columns)):
        x = field_amplitudes
        z = df[col].values
        y = np.full_like(x, y_index)

        color = colormap(y_index / num_lines)  # выбираем цвет из colormap

        if len(x) >= 4:
            x_smooth = np.linspace(x.min(), x.max(), 300)
            z_smooth = make_interp_spline(x, z, k=3)(x_smooth)
            y_smooth = np.full_like(x_smooth, y_index)

            ax.plot(x_smooth, y_smooth, z_smooth, linewidth=1.5, color=color)

            verts = list(zip(x_smooth, z_smooth))
            polys.append((verts, color))
        else:
            ax.plot(x, y, z, linewidth=1.5, color=color)
            verts = list(zip(x, z))
            polys.append((verts, color))

    for (verts, color), y_index in zip(polys, y_positions):
        poly = PolyCollection([verts], facecolors=[color], alpha=0.2)
        ax.add_collection3d(poly, zs=[y_index], zdir='y')

    ax.set_xlabel('Field Amplitude')
    ax.set_ylabel('Frequency (cm⁻¹)')
    ax.set_zlabel('Energy')

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([str(freq) for freq in freq_values])

    fig.tight_layout(pad=2)

    output_folder = os.path.join(os.path.dirname(filepath), "graphs")
    os.makedirs(output_folder, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(filepath))[0] + "_waterfall.png"
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=300)
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
