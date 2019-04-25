import os
from PIL import Image

fill_color = ''
folder_path = 'Planes'
new_path = 'Processed Planes'
for subdir, dirs, files in os.walk(folder_path):
    subpath = os.path.join(new_path, subdir.split('/')[-1])
    if not os.path.exists(subpath):
        os.makedirs(subpath)
    for i, file in enumerate(sorted(files)):
        try:
            img = Image.open(os.path.join(subdir, file)).convert('RGB')
            new_file = os.path.join(subpath, '{}.jpg'.format(i))
            img.save(new_file)
        except Exception as e:
            print(e)
