from PIL import Image
import imagehash
import os
import matplotlib.pyplot as plt

w_hash = {}

folder_path = 'Planes'
for subdir, dirs, files in os.walk(folder_path):
    for i, f in enumerate(files):
        file = os.path.join(subdir, f)
        try:
            img = Image.open(file)

            whash = imagehash.whash(img)
            if whash not in w_hash:
                w_hash[imagehash.whash(img)] = file
            else:
                fig, axes = plt.subplots(2, 1)
                axes[0].imshow(img)
                axes[1].imshow(Image.open(w_hash[whash]))
                plt.show()
                inp = input('Duplicates?')
                if inp == 'y':
                    os.remove(file)
                plt.close()


        except Exception as e:
            print(e, file)

