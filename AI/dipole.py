import os
from concurrent.futures import ProcessPoolExecutor
import MDAnalysis as mda

def calculate_dipole_for_frame(frame_info):
    """Функция для вычисления дипольного момента для одного кадра."""
    ts, universe, output_file = frame_info
    universe.trajectory[ts]
    atoms = universe.select_atoms("all")
    positions = atoms.positions
    charges = atoms.charges
    dipole_moment = sum(charge * position for charge, position in zip(charges, positions))
    
    with open(output_file, 'a') as f:
        f.write(f"Frame {ts}: {dipole_moment}\n")
    return ts  # Возвращаем номер обработанного кадра

def process_directory(directory, num_workers=2):
    """Обрабатывает одну папку с файлами."""
    psf_file = os.path.join(directory, "trp_1.psf")
    dcd_file = os.path.join(directory, "runned.dcd")
    output_file = os.path.join(directory, "dipole_moments.dat")
    
    if not os.path.exists(psf_file) or not os.path.exists(dcd_file):
        print(f"Пропускаем папку {directory}, отсутствуют необходимые файлы.")
        return

    print(f"Обрабатываем папку: {directory}")
    
    # Загрузка вселенной
    universe = mda.Universe(psf_file, dcd_file)
    
    # Формирование информации для обработки
    frames_info = [(ts, universe, output_file) for ts in range(len(universe.trajectory))]

    # Использование ProcessPoolExecutor для обработки кадров
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for ts in executor.map(calculate_dipole_for_frame, frames_info):
            print(f"Обработан кадр: {ts}")

def main():
    """Обрабатывает все папки в текущей директории."""
    base_directory = os.getcwd()
    num_workers = 4  # Укажите число потоков

    for subdir in [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]:
        process_directory(subdir, num_workers=num_workers)

if __name__ == "__main__":
    main()
