import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_waterfall_3d(filepath):
    df = pd.read_excel(filepath)

    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    aa_name = base_name.split('_')[0]

    # Первый столбец — Field Amplitudes
    field_amplitudes = df.iloc[:, 1].values

    # Частоты и названия столбцов POTENTIAL
    potential_columns = [col for col in df.columns if col.endswith("POTENTIAL") and col.startswith(aa_name)]
    freq_values = [int(col.replace(aa_name, '').replace("POTENTIAL", '')) for col in potential_columns]

    # Сортировка по частоте
    freq_and_cols = sorted(zip(freq_values, potential_columns))
    freq_values, sorted_columns = zip(*freq_and_cols)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Частоты — по оси Z (в глубину), равномерно
    z_positions = range(len(sorted_columns))

    for z_index, (freq, col) in zip(z_positions, zip(freq_values, sorted_columns)):
        energy_values = df[col].values
        ax.plot(field_amplitudes, energy_values, zs=z_index, zdir='z', label=str(freq), linewidth=1.5)

    # Подписи и оси
    ax.set_xlabel('Field Amplitude')
    ax.set_ylabel('Energy')
    ax.set_zlabel('Frequency (cm⁻¹)')

    # Деления оси Z — без единиц измерения
    ax.set_zticks(z_positions)
    ax.set_zticklabels([str(freq) for freq in freq_values])

    # Легенда
    ax.legend(title="Frequencies", loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')

    # Компактное размещение
    fig.tight_layout(pad=2)

    # Сохранение
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
    # Укажи путь к папке с Excel-файлами
    folder = os.getcwd()
    process_folder(folder)
