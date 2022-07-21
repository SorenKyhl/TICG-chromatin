import multiprocessing
import os
import os.path as osp

import hicstraw
import numpy as np
import pandas as pd
from seq2contact import DiagonalPreprocessing, plot_matrix


def download_contactmap_straw(filename, chrom, start, end, resolution):
    '''
    loads Hi-C contact map using straw https://github.com/aidenlab/straw/wiki
    uses Knight-Ruiz matrix balancing (s.t. contact map is symmetric)
    see here for common files https://www.aidenlab.org/data.html
    example input for knight-rubin (KR) normalized chromosome 2, 0-130Mbp at 100kbp resolution.
    raw_data = straw.straw("KR", filename, "2:0:129999999", "2:0:129999999", "BP", 100000)
    '''
    basepairs = f"{chrom}:{start}:{end}"
    result = hicstraw.straw("observed", "NONE", filename, basepairs, basepairs, "BP", resolution)

    m = int((end - start) / resolution)
    hic = np.zeros((m+1, m+1))
    for row in result:
        i = int((row.binX - start) / resolution)
        j = int((row.binY - start) / resolution)
        try:
            hic[i, j] = row.counts
            hic[j, i] = row.counts
        except Exception as e:
            print(e)
            print(row.binX, row.binY, row.counts, i, j)

    hic /= np.max(hic)

    return hic


def import_contactmap_straw(sample_folder, filename, chrom=2, start=22000000, end=60575000, resolution=25000):
    hic = download_contactmap_straw("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic", chrom, start, end, resolution)
    m, _ = hic.shape

    if not osp.exists(sample_folder):
        os.mkdir(sample_folder, mode = 0o755)

    with open(osp.join(sample_folder, 'import.log'), 'w') as f:
        f.write(f'{filename}\nchrom={chrom}\nstart={start}\nend={end}\nresolution={resolution}\nbeads={m}')


    np.save(osp.join(sample_folder, 'y.npy'), hic)
    print(f'{sample_folder} done')

def main():
    dir = '/home/erschultz/sequences_to_contact_maps'
    dataset='dataset_09_21_21'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

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
            sampleFolder = osp.join(data_folder, 'samples', f'sample{i}')
            mapping.append((sampleFolder, "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic", chromosome, start, end, resolution))
            i += 1

    with multiprocessing.Pool(6) as p:
        p.starmap(import_contactmap_straw, mapping)

def main2():
    # dir = '/home/erschultz/sequences_to_contact_maps'
    dir = '/project2/depablo/erschultz'
    dataset='dataset_07_20_22'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

    chromosome='4'
    start_mb=60
    resolution=5000
    filename="https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic"
    # filename='https://www.encodeproject.org/files/ENCFF718AWL/@@download/ENCFF718AWL.hic'
    hic = hicstraw.HiCFile(filename)
    print(hic.getGenomeID())
    print(hic.getResolutions())
    chr_mat = hic.getMatrixZoomData(chromosome, chromosome, "observed", "KR", "BP", resolution)
    for chrom in hic.getChromosomes():
        print(chrom.name, chrom.length)
    for i, end_mb in enumerate([65, 70, 80, 90, 110, 130, 150, 190]):
        start = start_mb * 10**6
        end = end_mb * 10**6
        m = (end - start) / resolution
        print(i+1, start, end, m)

        y = chr_mat.getRecordsAsMatrix(start, end, start, end)
        m, _ = y.shape

        sample_folder = osp.join(data_folder, 'samples', f'sample{i+1}')
        if not osp.exists(sample_folder):
            os.mkdir(sample_folder, mode = 0o755)

        with open(osp.join(sample_folder, 'import.log'), 'w') as f:
            f.write(f'{filename}\nchrom={chromosome}\nstart={start}\nend={end}\nresolution={resolution}\nbeads={m}')

        np.save(osp.join(sample_folder, 'y.npy'), y)

def main3():
    dir = '/home/erschultz/sequences_to_contact_maps'
    # dir = '/project2/depablo/erschultz'
    dataset='dataset_07_20_22'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

    chromosome='4'
    start_mb=60
    resolution=5000
    filename="https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic"
    filename='https://www.encodeproject.org/files/ENCFF718AWL/@@download/ENCFF718AWL.hic'

    # set up for multiprocessing
    mapping = []
    for i, m in enumerate([512, 1024, 2048, 4096, 8192, 16384]):
        start = start_mb * 10**6
        end = start + (m-1) * resolution
        print(f'i={i+1}', start, end, m)
        sample_folder = osp.join(data_folder, 'samples', f'sample{i+1}')
        mapping.append((sample_folder, filename, chromosome, start, end, resolution))


    with multiprocessing.Pool(10) as p:
        p.starmap(import_contactmap_straw, mapping)


if __name__ == '__main__':
    main3()
