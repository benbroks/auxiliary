import sys 
import os
import random
import shutil

import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from auxiliary_partition.config import raw_dir, utk_dir, TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT

def remove_og_dir():
    try:
        shutil.rmtree(raw_dir/'train')
        shutil.rmtree(raw_dir/'validation')
        shutil.rmtree(raw_dir/'test')
    except:
        print("Directories already deleted.")

def create_dirs():
    os.system('mkdir {dir}'.format(dir=raw_dir/'train'))
    os.system('mkdir {dir}'.format(dir=raw_dir/'validation'))
    os.system('mkdir {dir}'.format(dir=raw_dir/'test'))

def randomly_generate_dirs():
    for f in os.listdir(utk_dir):
        r = random.random()
        if r < TRAIN_SPLIT:
            new_dir = raw_dir / 'train'
        elif r < TRAIN_SPLIT + VALIDATION_SPLIT:
            new_dir = raw_dir / 'validation'
        else:
            new_dir = raw_dir / 'test'
        command = "cp {og_fp} {new_fp}".format(
            og_fp = utk_dir / f,
            new_fp = new_dir / f, 
        )
        os.system(command)

if __name__ == "__main__":
    remove_og_dir()
    create_dirs()
    randomly_generate_dirs()

    