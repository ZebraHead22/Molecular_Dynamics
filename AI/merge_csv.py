import os
import pandas as pd
import re


"""
Так как мы обрабатываем большие файлы в два подхода, то этот скрипт
нужен для того, чтобы слить 2 .csv файла воедино
"""
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def merge_csv_files(input_dir, output_file):
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and f != os.path.basename(output_file)]
    csv_files.sort(key=extract_number)
    
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path, sep=';', dtype=str)
        dataframes.append(df)
    
    merged_df = pd.concat(dataframes, axis=1)
    
    # Получение списка столбцов, отсортированного по извлеченным числам
    def extract_number_from_col(col_name):
        return int(''.join(filter(str.isdigit, col_name)))
    
    sorted_columns = sorted(merged_df.columns, key=extract_number_from_col)
    
    # Создание нового DataFrame в нужном порядке
    sorted_merged_df = merged_df[sorted_columns]
    
    print(sorted_merged_df)
    sorted_merged_df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    print(f"Merged CSV saved as: {output_file}")

if __name__ == "__main__":
    input_directory = os.getcwd()
    output_csv = os.path.join(input_directory, "merged_peaks.csv")
    
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    merge_csv_files(input_directory, output_csv)