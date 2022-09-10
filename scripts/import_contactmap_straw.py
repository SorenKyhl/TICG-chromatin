import multiprocessing
import os
import os.path as osp

import bioframe  # https://github.com/open2c/bioframe
import hicstraw  # https://github.com/aidenlab/straw
import numpy as np
import pandas as pd


def download_contactmap_straw(hic_filename, chrom, start, end, resolution, norm):
    basepairs = f"{chrom}:{start}:{end}"
    result = hicstraw.straw("observed", norm, hic_filename, basepairs, basepairs, "BP", resolution)

    m = int((end - start) / resolution)
    y_arr = np.zeros((m+1, m+1))
    for row in result:
        i = int((row.binX - start) / resolution)
        j = int((row.binY - start) / resolution)
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
    # print(f'{sample_folder} done')

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
    dir = '/home/erschultz/sequences_to_contact_maps'
    # dir = '/project2/depablo/erschultz'
    dataset='dataset_07_20_22'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

    start=60000000
    resolution=5000
    norm = 'KR'
    chromosome=4
    filename="https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic"
    filename='https://www.encodeproject.org/files/ENCFF718AWL/@@download/ENCFF718AWL.hic' #GM12878

    # set up for multiprocessing
    mapping = []
    for i, m in enumerate([512, 1024, 2048, 4096]):
        # chromsizes = bioframe.fetch_chromsizes('hg38')
        # end = chromsizes[f'chr{chromosome}']
        end = start + resolution * (m-1)
        # m = (end - start) / resolution + 1
        print(f'i={i+11}: m={m}')
        sample_folder = osp.join(data_folder, 'samples', f'sample{i+11}')
        mapping.append((sample_folder, filename, chromosome, start, end, resolution, norm))

    with multiprocessing.Pool(10) as p:
        p.starmap(import_contactmap_straw, mapping)


if __name__ == '__main__':
    main()
