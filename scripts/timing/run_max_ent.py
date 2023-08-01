import multiprocessing as mp
import sys

import numpy as np

sys.path.append('/home/erschultz/TICG-chromatin')
from max_ent import fit, setup_config


def rescale_bl_vb(m):
    # gaussian renormalization
    # reference m=512, bl=140, vb=130,000
    ratio = 512/m
    bl = 140 * ratio**(0.5)
    vb = 130000 * ratio
    return bl, vb

def main():
    m_list = [256, 512, 1024, 2560, 5120]
    m_list = [256, 512, 1024, 2560]
    bl_list = []
    vb_list= []
    for m in m_list:
        bl, vb = rescale_bl_vb(m)
        bl_list.append(bl)
        vb_list.append(vb)
    bl_list = [round(i) for i in bl_list]
    print(bl_list)
    print(vb_list)

    for j, m in enumerate(m_list):
        print(m)
        dataset = dataset = f'timing_analysis/{m}'
        samples = list(range(1, 16))
        mapping = []
        for i in samples:
            mapping.append((dataset, i, 'samples', bl_list[j], 0.03, vb_list[j]))
        print(len(mapping))
        print(mapping)

        with mp.Pool(15) as p:
            p.starmap(fit, mapping)



if __name__ == '__main__':
    main()
