import os
import os.path as osp
import sys

import straw
import pandas as pd
import numpy as np

# ensure that I can find contact_map
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)
from contact_map import *

paths = ['/home/erschultz/sequences_to_contact_maps',
        '/home/eric/Research/sequences_to_contact_maps',
        'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps']
for p in paths:
    if osp.exists(p):
        sys.path.insert(1, p)

from neural_net_utils.utils import diagonal_preprocessing, generateDistStats


def import_contactmap_straw(filename, chrom=2, start=35000000, end=60575000, resolution=25000):
#def import_contactmap_straw(filename, chrom=2, start=22000000, end=60575000, resolution=25000):
    '''
    loads Hi-C contact map using straw https://github.com/aidenlab/straw/wiki
    uses Knight-Ruiz matrix balancing (s.t. contact map is symmetric)
    see here for common files https://www.aidenlab.org/data.html
    example input for knight-rubin (KR) normalized chromosome 2, 0-130Mbp at 100kbp resolution.
    raw_data = straw.straw("KR", filename, "2:0:129999999", "2:0:129999999", "BP", 100000)
    '''
    basepairs = ":".join((str(chrom), str(start), str(end)))
    raw_data = straw.straw("KR", filename, basepairs, basepairs, "BP", resolution)
    raw_data = np.array(raw_data)
    # raw data is a map between locus pairs and reads -- need to pivot into a matrix.
    df = pd.DataFrame(raw_data.transpose(), columns=['locus1 [bp]', 'locus2 [bp]', 'reads'])
    pivoted = df.pivot_table(values='reads', index='locus1 [bp]', columns='locus2 [bp]')
    xticks = np.array(pivoted.columns) # in base pairs
    filled = pivoted.fillna(0)
    trans = filled.transpose()
    hic = np.array(filled + trans)
    # normalize so rows and columns sum to 1
    for i, row in enumerate(hic):
        # row normalization:
        #hic[i] /= sum(row)
        # diag normalization
        hic[i] /= hic[i,i]
        #hic[i] /= np.mean(hic.diagonal())
    return hic, xticks

def main():
    dataFolder ='/project2/depablo/erschultz/dataset_09_21_21'
    # dataFolder='dataset_09_21_21'
    os.mkdir(dataFolder, mode = 0o755)
    os.mkdir(osp.join(dataFolder, 'samples'), mode = 0o755)
    sample = 'sample1'

    sampleFolder = osp.join(dataFolder, 'samples', sample)
    os.mkdir(sampleFolder, mode = 0o755)

    ofile = osp.join(sampleFolder, 'y.npy')
    hic, xticks = import_contactmap_straw("https://s3.amazonaws.com/hicfiles/hiseq/degron/untreated/unsynchronized/combined.hic")
    print(hic.shape)

    np.save(osp.join(sampleFolder, 'y.npy'), hic)

    plotContactMap(hic, ofile = osp.join(sampleFolder, 'y.png'), vmax = 'mean')

    meanDist = generateDistStats(hic)
    y_diag_instance = diagonal_preprocessing(hic, meanDist)
    plotContactMap(y_diag_instance, ofile = osp.join(sampleFolder, 'y_diag_instance.png'), vmax = 'max')
    np.save(osp.join(sampleFolder, 'y_diag_instance.npy'), y_diag_instance)

if __name__ == '__main__':
    main()
