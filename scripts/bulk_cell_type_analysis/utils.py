import os
import os.path as osp
import sys

import numpy as np
from pylib.utils.plotting_utils import plot_matrix
from pylib.utils.utils import load_import_log

sys.path.append('/home/erschultz')
from sequences_to_contact_maps.scripts.load_utils import load_Y


def get_samples(dataset):
    samples_dir = osp.join('/home/erschultz/', dataset, 'samples')
    samples = os.listdir(samples_dir)

    target_import_log = load_import_log(osp.join(samples_dir, 'sample8'))

    final_samples = []
    cell_lines = []
    for s in samples:
        if s.startswith('sample'):
            s_id = int(s[6:])
            s_dir = osp.join(samples_dir, s)
            import_log = load_import_log(s_dir)
            keep = True
            for k in ['chrom', 'start', 'end', 'resolution']:
                if target_import_log[k] != import_log[k]:
                    keep = False
            if keep:
                final_samples.append(s_id)
                cell_lines.append(import_log['cell_line'])

    print(final_samples, cell_lines)
    return final_samples, cell_lines


def loci_to_coords(loci, import_log, mode='b'):
    '''Convert loci tuple to genomic coords.'''
    chrom = int(import_log['chrom'])
    if mode == 'mb':
        start = import_log['start_mb']
        resolution = import_log['resolution_mb']
    elif mode == 'b':
        start = import_log['start']
        resolution = import_log['resolution']

    loci[1] += 1
    loci = [start + resolution*i for i in loci]

    if mode == 'mb':
        loci = [np.round(i, 2) for i in loci]

    return f'Chr{chrom}:{loci[0]}-{loci[1]}'


def test_loci_to_coords():
    loci = [0,1]
    import_log = load_import_log('/home/erschultz/dataset_12_06_23/samples/sample451')
    coords = loci_to_coords(loci, import_log, mode='b')
    print(coords)



if __name__ == '__main__':
    get_samples('dataset_12_06_23')
    # test_loci_to_coords()
