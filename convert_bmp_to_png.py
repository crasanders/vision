import os
from PIL import Image

folder_path = 'VET'
new_path = 'VET_pngs'
for subdir, dirs, files in os.walk(folder_path):
    for i, file in enumerate(files):
        if file.endswith('.bmp'):
            img = Image.open(os.path.join(subdir, file)).convert('RGB')
            cat = subdir.split('/')[1]
            file_name = file.split('.')[0]
            new_file = os.path.join(new_path, cat, file_name + '.png')
            img.save(new_file)