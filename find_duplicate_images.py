from PIL import Image
import imagehash
import os

a_hash = {}
p_hash = {}
d_hash = {}
w_hash = {}

folder_path = '/Users/craigsanders/PycharmProjects/vision/VET_pngs/Cars'
for subdir, dirs, files in os.walk(folder_path):
    for i, file in enumerate(files):
        try:
            img = Image.open(os.path.join(subdir, file))

            a_hash[imagehash.average_hash(img)] = file
            p_hash[imagehash.phash(img)] = file
            d_hash[imagehash.dhash(img)] = file
            w_hash[imagehash.whash(img)] = file

        except Exception as e:
            print(e)

folder_path = '/Users/craigsanders/Desktop/Processed Cars'
for subdir, dirs, files in os.walk(folder_path):
    for i, file in enumerate(files):
        try:
            f = os.path.join(subdir, file)
            img = Image.open(f)

            ahash = imagehash.average_hash(img)
            phash = imagehash.phash(img)
            dhash = imagehash.dhash(img)
            whash = imagehash.whash(img)

            if ahash in a_hash:
                print('{} is a duplicate of {}'.format(f, a_hash[ahash]))
            elif phash in p_hash:
                print('{} is a duplicate of {}'.format(f, p_hash[phash]))
            elif dhash in d_hash:
                print('{} is a duplicate of {}'.format(f, d_hash[dhash]))
            elif whash in w_hash:
                print('{} is a duplicate of {}'.format(f, w_hash[whash]))

        except Exception as e:
            print(e)