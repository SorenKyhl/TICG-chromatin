import multiprocessing
import os
import os.path as osp

import bioframe  # https://github.com/open2c/bioframe
import hicstraw  # https://github.com/aidenlab/straw
import numpy as np
import pandas as pd

HG19_BAD_REGIONS = {1:'0-3,120-150',
                    2:'85-97',
                    3:'90-95',
                    4:'48-54',
                    5:'45-50,67-72',
                    6:'57-65',
                    7:'55-77',
                    8:'44-48',
                    9:'37-72',
                    10:'37-52',
                    11:'50-55',
                    12:'34-39',
                    13:'0-21',
                    14:'0-22',
                    15:'0-22',
                    16:'34-47',
                    17:'21-26',
                    18:'14-19',
                    19:'24-29',
                    20:'25-30',
                    21:'0-15,46-48',
                    22:'0-20'}

ALL_FILES = [
            "https://hicfiles.s3.amazonaws.com/hiseq/imr90/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/agar/HIC030.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/dilution/HIC034.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/hmec/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/nhek/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/kbm7/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/huvec/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/hela/in-situ/combined.hic",
            "https://hicfiles.s3.amazonaws.com/hiseq/hap1/in-situ/combined.hic"
            ]
            # "https://www.encodeproject.org/files/ENCFF675SJE/@@download/ENCFF675SJE.hic" hg38 A549
            # "https://www.encodeproject.org/files/ENCFF177TYX/@@download/ENCFF177TYX.hic"

ALL_FILES_NO_GM12878 = [f for f in ALL_FILES if 'gm12878' not in f]

def intersect(region, bad_region):
    # region/2 is a tuple
    if region[0] >= bad_region[0] and region[0] < bad_region[1]:
        return True
    if region[1] > bad_region[0] and region[1] < bad_region[1]:
        return True
    if bad_region[0] >= region[0] and bad_region[0] < region[1]:
        return True

    return False

def import_contactmap_straw(sample_folder, hic_filename, chrom, start,
                            end, resolution, norm = 'NONE'):
    basepairs = f"{chrom}:{start}:{end}"
    print(basepairs)
    result = hicstraw.straw("observed", norm, hic_filename, basepairs, basepairs, "BP", resolution)
    hic = hicstraw.HiCFile(hic_filename)

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
        print(f'{sample_folder} had no reads')
        return
    else:
        y_arr /= np.max(y_arr)


    m, _ = y_arr.shape

    if not osp.exists(sample_folder):
        os.mkdir(sample_folder, mode = 0o755)

    with open(osp.join(sample_folder, 'import.log'), 'w') as f:
        if isinstance(chrom, str):
            chrom = chrom.strip('chr')
        f.write(f'{hic_filename}\nchrom={chrom}\nstart={start}\nend={end}\n')
        f.write(f'resolution={resolution}\nbeads={m}\nnorm={norm}\n')
        f.write(f'genome={hic.getGenomeID()}')

    np.save(osp.join(sample_folder, 'y.npy'), y_arr)
    print(f'{sample_folder} done')

def single_cell_import():
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

def import_wrapper(data_folder, filename_list, resolution, norm, m,
                    i, ref_genome, chroms):
    if isinstance(filename_list, str):
        filename_list = [filename_list]
    chromsizes = bioframe.fetch_chromsizes(ref_genome)
    mapping = []
    for filename in filename_list:
        for chromosome in chroms:
            start_mb = 0
            start = start_mb * 1000000
            end = start + resolution * m
            end_mb = end / 1000000
            while end < chromsizes[f'chr{chromosome}']:
                for region in HG19_BAD_REGIONS[chromosome].split(','):
                    region = region.split('-')
                    region = [int(d) for d in region]
                    if intersect((start_mb, end_mb), region):
                        start_mb = region[1] # skip to end of bad region
                        start_mb += np.random.choice(np.arange(6)) # add random shift
                        break
                else:
                    print(f'i={i}: chr{chromosome} {start_mb}-{end_mb}')
                    sample_folder = osp.join(data_folder, 'samples', f'sample{i}')
                    mapping.append((sample_folder, filename, chromosome, start,
                                    end, resolution, norm))
                    i += 1
                    start_mb = end_mb

                start = int(start_mb * 1000000)
                end = start + resolution * m
                end_mb = end / 1000000

    with multiprocessing.Pool(15) as p:
        p.starmap(import_contactmap_straw, mapping)


def dataset_01_26():
    dir = '/home/erschultz'
    dataset='dataset_01_26_23'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

    resolution=10000
    norm = 'NONE'
    filename="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104333/suppl/GSE104333_Rao-2017-treated_6hr_combined_30.hic"
    m = 512*5

    # slices = [(1, 15), (1, 166), (2, 110), (3, 130), (4, 140), (7, 89)]
    slices = [(5, 137), (6, 80), (10, 63), (13, 30)]

    # set up for multiprocessing
    mapping = []
    start_sample=89
    for i, (chromosome, start_mb) in enumerate(slices):
        start = start_mb * 1000000
        end = start + resolution * m
        end_mb = end / 1000000
        print(f'i={i+start_sample}: chr{chromosome} {start_mb}-{end_mb}')
        sample_folder = osp.join(data_folder, 'samples', f'sample{i+start_sample}')
        mapping.append((sample_folder, filename, chromosome, start, end, resolution, norm))

    with multiprocessing.Pool(10) as p:
        p.starmap(import_contactmap_straw, mapping)

def dataset_11_14():
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
    filename='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104333/suppl/GSE104333_Rao-2017-treated_6hr_combined_30.hic'
    m = 1024*5

    # slices = [(2, 17.6)]
    # slices = [(1, 15), (1, 150), (2, 5), (2, 135), (3, 5), (3, 110), (4, 60),
    #             (4, 115), (5, 75), (5, 120), (6, 4), (6, 100), (7, 6), (7, 77),]
    slices = [            (8, 90), (9, 75), (10, 55), (11, 56), (12, 45), (13, 22),]
    #             # (14, 35), (18, 20)]

    # set up for multiprocessing
    mapping = []
    start_sample=2017
    for i, (chromosome, start_mb) in enumerate(slices):
        start = start_mb * 1000000
        end = start + resolution * m
        end_mb = end / 1000000
        print(f'i={i+start_sample}: chr{chromosome} {start_mb}-{end_mb}')
        sample_folder = osp.join(data_folder, 'samples', f'sample{i+start_sample}')
        mapping.append((sample_folder, filename, chromosome, start, end, resolution, norm))

    with multiprocessing.Pool(10) as p:
        p.starmap(import_contactmap_straw, mapping)

def single_experiment_dataset(filename, dataset, resolution, m,
                                norm='NONE', i=1, ref_genome='hg19',
                                chroms=range(1,23)):
    dir = '/home/erschultz'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

    import_wrapper(data_folder, filename, resolution, norm, m, i, ref_genome, chroms)

def mixed_experimental_dataset(dataset, resolution, m, norm='NONE',
                                i=1, ref_genome='hg19',
                                chroms=range(1,23), files=ALL_FILES):
    dir = '/home/erschultz'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

    import_wrapper(data_folder, files, resolution, norm, m, i , ref_genome, chroms)

def Su2020imr90():
    sample_folder = '/home/erschultz/Su2020/samples/sample3'
    filename='https://hicfiles.s3.amazonaws.com/hiseq/imr90/in-situ/combined.hic'
    filename='/home/erschultz/Su2020/ENCFF281ILS.hic'

    resolution = 10000
    start = 14000001
    end = 46700001
    # end = start + 1024*5*resolution
    import_contactmap_straw(sample_folder, filename, 'chr21', start, end, resolution, 'NONE')



def download_techinical_replicates():
    dir = '/home/erschultz'
    # dir = '/project2/depablo/erschultz'
    dataset='dataset_test'
    data_folder = osp.join(dir, dataset)
    if not osp.exists(data_folder):
        os.mkdir(data_folder, mode = 0o755)
    if not osp.exists(osp.join(data_folder, 'samples')):
        os.mkdir(osp.join(data_folder, 'samples'), mode = 0o755)

    resolution=10000
    norm = 'NONE'
    m = 1024*5
    chromosome, start_mb =(1, 15)
    files = ['ENCFF241RAY', 'ENCFF453DBX', 'ENCFF374EBH', 'ENCFF415XWQ',
            'ENCFF246DOF', 'ENCFF125FXX', 'ENCFF590VOM']

    # set up for multiprocessing
    mapping = []
    start_sample=2001
    for i, f in enumerate(files):
        filename=f"https://www.encodeproject.org/files/{f}/@@download/{f}.hic"
        start = start_mb * 1000000
        end = start + resolution * m
        end_mb = end / 1000000
        print(f'i={i+start_sample}: chr{chromosome} {start_mb}-{end_mb}')
        sample_folder = osp.join(data_folder, 'samples', f'sample{i+start_sample}')
        mapping.append((sample_folder, filename, chromosome, start, end, resolution, norm))

    with multiprocessing.Pool(10) as p:
        p.starmap(import_contactmap_straw, mapping)

def pool():
    dir = '/home/erschultz/dataset_test/samples'
    y_list = []
    f_list = []
    for i in [2001, 2002, 2003, 2004]:
        s_dir = osp.join(dir, f'sample{i}')
        y = np.load(osp.join(s_dir, 'y.npy'))
        y_list.append(y)
        with open(osp.join(s_dir, 'import.log'), 'r') as f:
            f_list.append(f.readline())
            rest = f.readlines()
    print(f_list)

    s_dir = osp.join(dir, 'sample2101')
    if not osp.exists(s_dir):
        os.mkdir(s_dir, mode = 0o755)
    y = np.array(y_list)
    y = np.sum(y, axis = 0)
    print(y.shape)
    np.save(osp.join(s_dir, 'y.npy'), y)
    with open(osp.join(s_dir, 'import.log'), 'w') as f:
        for l in f_list:
            f.write(l)
        for l in rest:
            f.write(l)

    y_list = []
    f_list = []
    for i in [2005, 2006, 2007]:
        s_dir = osp.join(dir, f'sample{i}')
        y = np.load(osp.join(s_dir, 'y.npy'))
        y_list.append(y)
        with open(osp.join(s_dir, 'import.log'), 'r') as f:
            f_list.append(f.readline())
            rest = f.readlines()
    print(f_list)

    s_dir = osp.join(dir, 'sample2102')
    if not osp.exists(s_dir):
        os.mkdir(s_dir, mode = 0o755)
    y = np.array(y_list)
    y = np.sum(y, axis = 0)
    print(y.shape)
    np.save(osp.join(s_dir, 'y.npy'), y)
    with open(osp.join(s_dir, 'import.log'), 'w') as f:
        for l in f_list:
            f.write(l)
        for l in rest:
            f.write(l)


if __name__ == '__main__':
    # single_experiment_dataset("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
    #                             'dataset_02_04_23', 10000, 512*5)
    # single_experiment_dataset("https://hicfiles.s3.amazonaws.com/hiseq/imr90/in-situ/combined.hic",
    #                             'dataset_02_21_23', 10000, 512*5)
    # mixed_experimental_dataset('dataset_03_21', 10000, 512*5)
    # files = ["https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
            # "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104333/suppl/GSE104333_Rao-2017-untreated_combined_30.hic"]
    files = ["https://www.encodeproject.org/files/ENCFF177TYX/@@download/ENCFF177TYX.hic"]
    # mixed_experimental_dataset('dataset_04_05_23', 10000, 1024*5, files = files, i=263)
    # mixed_experimental_dataset('dataset_04_06', 10000, 1024*5)
    # mixed_experimental_dataset('dataset_04_07', 25000, 1024*4)
    # single_experiment_dataset("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                                # 'dataset_04_09_23', 25000, 512*4)
    # single_experiment_dataset("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                                # 'dataset_04_10_23', 10000, 1024*5)
    # single_experiment_dataset("https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic",
                                # 'dataset_test', 25000, 1024*4, i=5001, chroms=[2])
    # single_experiment_dataset("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104333/suppl/GSE104333_Rao-2017-untreated_combined_30.hic",
    #                             'dataset_HCT116', 10000, 512*5, i=10, chroms=[2])
    # single_experiment_dataset("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104333/suppl/GSE104333_Rao-2017-treated_6hr_combined_30.hic",
                                # 'dataset_HCT116_RAD21_KO', 10000, 512*5, i=10, chroms=[2])

    Su2020imr90()
