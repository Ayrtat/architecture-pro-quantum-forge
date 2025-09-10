import os
import json

# Загружаем термины из JSON файла
with open('terms_map.json', 'r') as f:
    terms_map = json.load(f)

input_dir = './wowpedia'
output_dir = './knowledge_base'

# Создаем выходную директорию, если она не существует
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Проходим по всем файлам в входной директории
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Заменяем имена на новые значения из terms_map
        for key, value in terms_map.items():
            content = content.replace(key, value)
        
        # Записываем измененный контент в новый файл
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as file:
            file.write(content)
