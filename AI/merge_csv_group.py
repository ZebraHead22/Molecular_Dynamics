import os
import pandas as pd
import re

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def merge_csv_files(input_dir, output_file, frequency_column=None):
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and f != os.path.basename(output_file)]
    csv_files.sort(key=extract_number)
    
    all_frequencies = []
    file_data = []
    
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        try:
            df = pd.read_csv(file_path, sep=';', dtype=str)
            df.columns = df.columns.str.strip().str.lower()
            
            # Автоматическое определение столбца с частотами
            if frequency_column:
                freq_col = frequency_column.lower()
                if freq_col not in df.columns:
                    print(f"Файл {file} пропущен: столбец '{frequency_column}' не найден.")
                    continue
            else:
                numeric_cols = []
                for col in df.columns:
                    if col == 'frequency':  # Проверяем стандартное имя
                        freq_col = col
                        break
                    try:
                        sample = df[col].dropna().iloc[0].replace(',', '.')
                        float(sample)
                        numeric_cols.append(col)
                    except:
                        continue
                else:
                    if numeric_cols:
                        freq_col = numeric_cols[0]
                    else:
                        print(f"Файл {file} пропущен: числовые столбцы не найдены.")
                        continue
            
            # Обработка данных
            freq_values = {}
            for _, row in df.iterrows():
                freq_str = row[freq_col].replace(',', '.')
                try:
                    freq = float(freq_str)
                except ValueError:
                    continue
                
                for col in df.columns:
                    if col != freq_col and pd.notna(row[col]):
                        value = row[col].strip()
                        freq_values[freq] = value
                        break
            file_data.append({
                'filename': file,
                'freq_value': freq_values
            })
            all_frequencies.extend(freq_values.keys())
        
        except Exception as e:
            print(f"Ошибка при чтении файла {file}: {e}")
            continue
    
    if not all_frequencies:
        pd.DataFrame().to_csv(output_file, sep=';', index=False)
        print(f"Merged CSV saved as: {output_file}")
        return
    
    # Группировка частот с допуском 1.5
    unique_freq = sorted(list(set(all_frequencies)))
    tolerance = 1.5
    groups = []
    if unique_freq:
        current_group = [unique_freq[0]]
        max_in_group = unique_freq[0]
        for freq in unique_freq[1:]:
            if freq <= max_in_group + tolerance:
                current_group.append(freq)
                if freq > max_in_group:
                    max_in_group = freq
            else:
                groups.append(current_group)
                current_group = [freq]
                max_in_group = freq
        groups.append(current_group)
    
    # Создание итоговых данных
    group_info = []
    for group in groups:
        group_min = min(group)
        group_max = max(group)
        mean_freq = sum(group) / len(group)
        group_info.append({
            'interval': (group_min - tolerance, group_max + tolerance),
            'mean_freq': mean_freq
        })
    
    group_info.sort(key=lambda x: x['mean_freq'])
    
    # Формирование строк CSV
    rows = []
    for group in group_info:
        row = {'frequency': f"{group['mean_freq']:.2f}"}
        for file_entry in file_data:
            matched = None
            min_diff = float('inf')
            for freq in file_entry['freq_value']:
                if group['interval'][0] <= freq <= group['interval'][1]:
                    diff = abs(freq - group['mean_freq'])
                    if diff < min_diff:
                        min_diff = diff
                        matched = file_entry['freq_value'][freq]
            row[file_entry['filename']] = matched
        rows.append(row)
    
    # Сохранение результата
    if rows:
        merged_df = pd.DataFrame(rows)
        columns = ['frequency'] + sorted(csv_files, key=extract_number)
        merged_df = merged_df.reindex(columns=columns)
        merged_df.to_csv(output_file, sep=';', index=False)
    else:
        pd.DataFrame().to_csv(output_file, sep=';', index=False)
    
    print(f"Merged CSV saved as: {output_file}")

if __name__ == "__main__":
    input_directory = os.getcwd()
    output_csv = os.path.join(input_directory, "merged_peaks.csv")
    
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    # Укажите имя вашего столбца с частотами, например: frequency_column='Freq'
    merge_csv_files(input_directory, output_csv, frequency_column=None)