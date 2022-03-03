import multiprocessing
import os
import os.path as osp

import numpy as np
import pandas as pd
from seq2contact import (diagonal_preprocessing, genomic_distance_statistics,
                         import, plotContactMap)

import straw


def download_contactmap_straw(filename, chrom, start, end, resolution):
    '''
    loads Hi-C contact map using straw https://github.com/aidenlab/straw/wiki
    uses Knight-Ruiz matrix balancing (s.t. contact map is symmetric)
    see here for common files https://www.aidenlab.org/data.html
    example input for knight-rubin (KR) normalized chromosome 2, 0-130Mbp at 100kbp resolution.
    raw_data = straw.straw("KR", filename, "2:0:129999999", "2:0:129999999", "BP", 100000)
    '''
    basepairs = ":".join((str(chrom), str(start), str(end)))
    raw_data = straw.straw("NONE", filename, basepairs, basepairs, "BP", resolution)
    raw_data = np.array(raw_data)
    # raw data is a map between locus pairs and reads -- need to pivot into a matrix.
    df = pd.DataFrame(raw_data.transpose(), columns=['locus1 [bp]', 'locus2 [bp]', 'reads'])
    pivoted = df.pivot_table(values='reads', index='locus1 [bp]', columns='locus2 [bp]')
    xticks = np.array(pivoted.columns) # in base pairs
    filled = pivoted.fillna(0)
    trans = filled.transpose()
    hic = np.array(filled + trans)
    print(hic)
    # normalize so rows and columns sum to 1
    for i, row in enumerate(hic):
        # row normalization:
        #hic[i] /= sum(row)
        # diag normalization
        hic[i] /= hic[i,i]
        #hic[i] /= np.mean(hic.diagonal())
    print(hic)
    return hic, xticks

def import_contactmap_straw(sample_folder, filename, chrom=2, start=22000000, end=60575000, resolution=25000):
    hic, xticks = download_contactmap_straw("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic", chrom, start, end, resolution)
    m, _ = hic.shape

    if not osp.exists(sample_folder):
        os.mkdir(sample_folder, mode = 0o755)

    with open(osp.join(sample_folder, 'import.log'), 'w') as f:
        f.write(f'{filename}\nchrom={chrom}\nstart={start}\nend={end}\nresolution={resolution}\nbeads={m}')

    np.save(osp.join(sample_folder, 'y.npy'), hic)
    plotContactMap(hic, ofile = osp.join(sample_folder, 'y.png'), vmax = 'mean')
    plotContactMap(hic, ofile = osp.join(sample_folder, 'y_max.png'), vmax = 'max')
    np.savetxt(osp.join(sample_folder, 'y.txt'), hic)

    meanDist = genomic_distance_statistics(hic)
    y_diag = diagonal_preprocessing(hic, meanDist)
    plotContactMap(y_diag, ofile = osp.join(sample_folder, 'y_diag.png'), vmax = 'max')
    np.save(osp.join(sample_folder, 'y_diag.npy'), y_diag)

def main():
    dir = 'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps'
    # dir ='/home/eric/sequences_to_contact_maps'
    # dir = '/project2/depablo/erschultz'
    dataset='dataset_09_21_21'
    dataFolder = osp.join(dir, dataset)
    if not osp.exists(dataFolder):
        os.mkdir(dataFolder, mode = 0o755)
    if not osp.exists(osp.join(dataFolder, 'samples')):
        os.mkdir(osp.join(dataFolder, 'samples'), mode = 0o755)

    i = 2
    start_end_resolution = [(22000000, 42000000, 5000), (100000000, 110000000, 5000), (120000000, 125000000, 5000),
                            (22000000, 42000000, 10000), (100000000, 115000000, 10000), (120000000, 130000000, 10000),
                            (22000000, 42000000, 25000), (100000000, 130000000, 25000), (130000000, 160000000, 25000),
                            (22000000, 42000000, 50000), (100000000, 130000000, 50000), (130000000, 160000000, 50000)]
    # set up for multiprocessing
    mapping = []
    for start, end, resolution in start_end_resolution:
        for chromosome in [2, 7]:
            m = (end - start) / resolution
            print(i)
            sampleFolder = osp.join(dataFolder, 'samples', f'sample{i}')
            mapping.append((sampleFolder, "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic", chromosome, start, end, resolution))
            i += 1

    with multiprocessing.Pool(6) as p:
        p.starmap(import_contactmap_straw, mapping)

def main2():
    dir = 'C:/Users/Eric/OneDrive/Documents/Research/Coding/sequences_to_contact_maps'
    # dir ='/home/eric/sequences_to_contact_maps'
    # dir = '/project2/depablo/erschultz'
    dataset='dataset_09_21_21'
    dataFolder = osp.join(dir, dataset)
    if not osp.exists(dataFolder):
        os.mkdir(dataFolder, mode = 0o755)
    if not osp.exists(osp.join(dataFolder, 'samples')):
        os.mkdir(osp.join(dataFolder, 'samples'), mode = 0o755)

    i = 5
    chromosome=7
    start=100000000
    end=110000000
    resolution=5000
    m = (end - start) / resolution
    print(m)
    sampleFolder = osp.join(dataFolder, 'samples', f'sample{i}')
    import_contactmap_straw(sampleFolder, "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic", chromosome, start, end, resolution)


if __name__ == '__main__':
    main2()
