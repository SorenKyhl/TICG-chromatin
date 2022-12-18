import multiprocessing
import os
import os.path as osp

import bioframe  # https://github.com/open2c/bioframe
import hicstraw  # https://github.com/aidenlab/straw
import numpy as np
import pandas as pd


def download_contactmap_straw(hic_filename, chrom, start, end, resolution, norm):
    basepairs = f"{chrom}:{start}:{end}"
    print(basepairs)
    result = hicstraw.straw("observed", norm, hic_filename, basepairs, basepairs, "BP", resolution)

    m = int((end - start) / resolution)
    y_arr = np.zeros((m, m))
    for row in result:
        i = int((row.binX - start) / resolution)
        j = int((row.binY - start) / resolution)
        if i >= m or j >= m:
            continue
        try:
            y_arr[i, j] = row.counts
            y_arr[j, i] = row.counts
        except Exception as e:
            print(e)
            print(row.binX, row.binY, row.counts, i, j)

    if np.max(y_arr) == 0:
        return None
    else:
        return y_arr / np.max(y_arr)


def import_contactmap_straw(sample_folder, hic_filename, chrom, start,
                            end, resolution, norm = 'NONE'):
    y_arr = download_contactmap_straw(hic_filename, chrom, start, end, resolution, norm)
    if y_arr is None:
        print(f'{sample_folder} had no reads')
        return

    m, _ = y_arr.shape

    if not osp.exists(sample_folder):
        os.mkdir(sample_folder, mode = 0o755)

    with open(osp.join(sample_folder, 'import.log'), 'w') as f:
        f.write(f'{hic_filename}\nchrom={chrom}\nstart={start}\nend={end}\n')
        f.write(f'resolution={resolution}\nbeads={m}\nnorm={norm}')

    np.save(osp.join(sample_folder, 'y.npy'), y_arr)
    print(f'{sample_folder} done')

def main2():
    dir = '/home/erschultz/sequences_to_contact_maps'
    # dir = '/project2/depablo/erschultz'
    dataset='single_cell_nagano_2017/samples'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)

    chromosome='10'
    start_mb=0
    resolution=50000
    m=2600
    start = start_mb * 10**6
    end = start + (m-1) * resolution

    # set up for multiprocessing
    mapping = []
    for file in os.listdir(data_folder):
        sample_folder = osp.join(data_folder, file)
        filename = osp.join(sample_folder, 'adj.hic')
        mapping.append((sample_folder, filename, chromosome, start, end, resolution))

    # print(mapping)
    with multiprocessing.Pool(15) as p:
        p.starmap(import_contactmap_straw, mapping)


def main():
    dir = '/home/erschultz'
    dataset='dataset_07_20_22'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

    resolution=10000
    norm = 'NONE'
    filename='https://www.encodeproject.org/files/ENCFF718AWL/@@download/ENCFF718AWL.hic' #GM12878
    m = 1024*5

    slices = [(10, 60)]


    # set up for multiprocessing
    mapping = []
    start_sample=2010
    for i, (chromosome, start_mb) in enumerate(slices):
        start = start_mb * 1000000
        end = start + resolution * m
        end_mb = end / 1000000
        print(f'i={i+start_sample}: chr{chromosome} {start_mb}-{end_mb}')
        sample_folder = osp.join(data_folder, 'samples', f'sample{i+start_sample}')
        mapping.append((sample_folder, filename, chromosome, start, end, resolution, norm))

    with multiprocessing.Pool(10) as p:
        p.starmap(import_contactmap_straw, mapping)

def main3():
    dir = '/home/erschultz'
    # dir = '/project2/depablo/erschultz'
    dataset='dataset_11_14_22'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

    resolution=10000
    norm = 'NONE'
    filename="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104333/suppl/GSE104333_Rao-2017-treated_6hr_combined_30.hic"
    m = 1024*5

    slices = [(2, 17.6)]
    # slices = [(2, 5), (2, 135), (3, 5), (3, 110), (4, 60),
    #             (4, 115), (5, 75), (5, 120), (6, 4), (6, 100), (7, 6), (7, 77),]
    #             # (8, 90), (9, 75), (10, 55), (11, 56), (12, 45), (13, 22),
    #             # (14, 35), (18, 20)]

    # set up for multiprocessing
    mapping = []
    start_sample=2016
    for i, (chromosome, start_mb) in enumerate(slices):
        start = start_mb * 1000000
        end = start + resolution * m
        end_mb = end / 1000000
        print(f'i={i+start_sample}: chr{chromosome} {start_mb}-{end_mb}')
        sample_folder = osp.join(data_folder, 'samples', f'sample{i+start_sample}')
        mapping.append((sample_folder, filename, chromosome, start, end, resolution, norm))

    with multiprocessing.Pool(10) as p:
        p.starmap(import_contactmap_straw, mapping)


if __name__ == '__main__':
    main3()
