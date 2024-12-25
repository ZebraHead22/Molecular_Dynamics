import os
from multiprocessing import Pool, cpu_count
import numpy as np
import MDAnalysis as mda

def calculate_dipole_for_frame(frame_info):
    """Вычисляет дипольный момент для заданного кадра."""
    frame, charges, positions = frame_info
    dipole = np.sum(charges[:, None] * positions, axis=0)  # q * r
    return frame, dipole

def process_directory(directory):
    """Обрабатывает одну папку с файлами."""
    psf_file = os.path.join(directory, "trp_1.psf")
    dcd_file = os.path.join(directory, "runned.dcd")
    velocities_file = os.path.join(directory, "velocities.dcd")  # Если потребуется
    
    # Проверяем наличие файлов
    if not (os.path.exists(psf_file) and os.path.exists(dcd_file)):
        print(f"Skipping {directory}: missing required files.")
        return

    # Загрузка файлов
    print(f"Processing directory: {directory}")
    u = mda.Universe(psf_file, dcd_file, velocities=velocities_file if os.path.exists(velocities_file) else None)
    charges = u.atoms.charges  # Заряды атомов
    
    # Подготовка данных для обработки
    frames_info = [
        (ts.frame, charges, u.atoms.positions) for ts in u.trajectory
    ]
    
    # Настройка многопроцессорной обработки
    # num_workers = cpu_count()
    NUM_WORKERS = 16
    results = []
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.imap_unordered(calculate_dipole_for_frame, frames_info)
    
    # Сохранение результатов
    output_file = os.path.join(directory, "dipole_moments.dat")
    with open(output_file, "w") as outfile:
        outfile.write("# Frame Dx Dy Dz\n")
        for frame, dipole in results:
            outfile.write(f"{frame} {dipole[0]:.6f} {dipole[1]:.6f} {dipole[2]:.6f}\n")
    print(f"Results saved to {output_file}")

def main():
    base_dir = "./data"  # Путь к основной папке, содержащей подкаталоги
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for subdir in subdirs:
        process_directory(subdir)

if __name__ == "__main__":
    main()
