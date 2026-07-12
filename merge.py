import os

# Напишите здесь расширения файлов, которые нужно собрать (например, ['.py', '.js', '.html', '.css'])
TARGET_EXTENSIONS = ['.py', '.txt'] 
OUTPUT_FILE = 'vsi_kod.txt'

with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk('.'):
        # Пропускаем системные папки, чтобы не собирать мусор
        if any(ignored in root for ignored in ['.git', '__pycache__', 'venv', '.idea']):
            continue
            
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in TARGET_EXTENSIONS and file != OUTPUT_FILE and file != 'merge.py':
                file_path = os.path.join(root, file)
                
                # Пишем красивый заголовок для каждого файла
                outfile.write(f"\n\n{'='*50}\n")
                outfile.write(f"ФАЙЛ: {file_path}\n")
                outfile.write(f"{'='*50}\n\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception:
                    outfile.write(f"[Не удалось прочитать файл {file}]\n")

print(f"Готово! Все файлы собраны в {OUTPUT_FILE}")
