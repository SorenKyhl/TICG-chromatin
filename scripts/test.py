import sys
import os
import os.path as osp

import numpy as np

def main():
    dir = "/project2/depablo/erschultz/dataset_08_24_21/samples"
    for file in os.listdir(dir):
        if file.startswith('sample'):
            if not osp.exists(osp.join(dir, file, 'seq0.txt')):
                print(file)

if __name__ == '__main__':
    main()
