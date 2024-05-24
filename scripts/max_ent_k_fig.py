import sys

import matplotlib.pyplot
import numpy as np
import scipy.stats as ss
from data_generation.modify_maxent import get_samples
from makeLatexTable import *

dataset = 'dataset_12_06_23'
samples, _ = get_samples(dataset, train=True, filter_cell_lines=['imr90'])
samples_list = samples[:10]
grid_root = 'optimize_grid_b_200_v_8_spheroid_1.5'


args = getArgs(data_folder = f'/home/erschultz/{dataset}',
                samples = samples_list)
args.experimental = True
args.verbose = True
args.bad_methods=['phi', 'grid200', '_spheroid_2.0']
args.convergence_definition = 'normal'
args.gnn_id = []
data, _ = load_data(args)

k_list = list(range(1, 16))
mean_arr = np.zeros(len(k_list))
std_arr = np.zeros(len(k_list))
for i, k in enumerate(k_list):
    print(f'k={k}')
    max_ent = f'{grid_root}-max_ent{k}'
    if max_ent in data[k]:
        max_ent_sccs = data[k][max_ent]['scc_var']
        print('\t', max_ent_sccs)
        mean_arr[i] = np.nanmean(max_ent_sccs)
        std_arr[i] = ss.sem(max_ent_sccs, nan_policy='omit')

plt.errorbar(k_list, mean_arr, std_arr, c='b')
plt.ylim([0,0.9])
plt.xlabel('$k$', fontsize=16)
plt.ylabel('SCC', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig('/home/erschultz/TICG-chromatin/figures/max_ent_k_fig.png')
plt.close()
# plt.show()
