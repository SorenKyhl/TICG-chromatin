import sys
import os
import os.path as osp

import numpy as np

def main():
    ids = set(range(1, 2001))
    dir = "/project2/depablo/erschultz/dataset_08_26_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            id = int(file[6:])
            ids.remove(id)

    print(ids)


if __name__ == '__main__':
    main()
