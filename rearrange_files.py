#%%
import os
from os.path import join
import numpy as np
from shutil import *

#%%
# DATA_DIR = r"C:\Users\Rajkumar\Downloads\ASL\dataset5"
DATA_DIR = r"D:/GWU/DATS-6303/project/archive/dataset5"
COLLATED_DIR = join(DATA_DIR, "collated")
#%%
if "collated" in os.listdir(DATA_DIR):
    rmtree(COLLATED_DIR)

os.mkdir(COLLATED_DIR)

subjects = os.listdir(DATA_DIR)
subjects = list(filter(lambda x: x != "collated", subjects))
max_len = 100000
for subject in subjects:
    subject_folder = f"{DATA_DIR}{os.path.sep}{subject}"
    char_folders = os.listdir(subject_folder)
    for char_ in char_folders:
        char_path = join(subject_folder, char_)
        collated_char_path = join(COLLATED_DIR, char_)
        if not char_ in os.listdir(COLLATED_DIR):
            os.mkdir(collated_char_path)
        img_fnames = os.listdir(char_path)
        for img_fname in img_fnames:
            fname = img_fname.split(".")[0]
            new_fname = fname + f"_{np.random.randint(0, max_len)}.png"
            copyfile(src=join(char_path, f"{fname}.png"), dst=join(collated_char_path, new_fname))










# %%
