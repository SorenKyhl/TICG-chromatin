import sys

import matplotlib.pyplot
import numpy as np
import scipy.stats as ss
from data_generation.modify_maxent import get_samples
from makeLatexTable import *

dataset = 'dataset_12_06_23'
samples, _ = get_samples(dataset, train=True, filter_cell_lines=['imr90'])
samples_list = samples[:10]
print(samples_list)
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

fig, ax = plt.subplots(1, 1)
ax.errorbar(k_list, mean_arr, std_arr, c='b')
ax.set_ylim([0,0.9])
ax.set_xlabel('$k$', fontsize=16)
ax.set_ylabel('SCC', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

delta = np.zeros(len(k_list))
delta[0] = np.NaN
prev = mean_arr[0]
for i, mean in enumerate(mean_arr[1:]):
    delta[i+1] = mean - prev
    prev = mean

plt.tight_layout()
plt.subplots_adjust(wspace=0.25)
plt.savefig('/home/erschultz/TICG-chromatin/figures/max_ent_k_fig.png')
plt.close()
# plt.show()


fig, axes = plt.subplots(1, 2)
fig.set_figheight(5.5)
fig.set_figwidth(12)
ax1, ax2 = axes


ax1.errorbar(k_list, mean_arr, std_arr, c='b')
ax1.set_ylim([0,0.9])
ax1.set_xlabel('$k$', fontsize=16)
ax1.set_ylabel('SCC', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)

delta = np.zeros(len(k_list))
delta[0] = np.NaN
prev = mean_arr[0]
for i, mean in enumerate(mean_arr[1:]):
    delta[i+1] = mean - prev
    prev = mean

ax2.plot(k_list, delta, c='b')
ax2.axhline(0.02, ls='--', c='k')
ax2.set_xticks(k_list)
ax2.set_xlabel('$k$', fontsize=16)
ax2.set_ylabel('$\Delta$SCC (k - k-1)', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.subplots_adjust(wspace=0.25)
plt.savefig('/home/erschultz/TICG-chromatin/figures/max_ent_k_fig_AB.png')
plt.close()
# plt.show()

fig, ax = plt.subplots(1, 1)

delta = np.zeros(len(k_list))
delta[0] = np.NaN
prev = mean_arr[0]
for i, mean in enumerate(mean_arr[1:]):
    delta[i+1] = mean - prev
    prev = mean

ax.plot(k_list, delta, c='b')
ax.axhline(0.01, ls='--', c='k')
ax.set_xticks(k_list)
ax.set_xlabel('$k$', fontsize=16)
ax.set_ylabel('$\Delta$SCC (k - k-1)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig('/home/erschultz/TICG-chromatin/figures/max_ent_k_fig_B2.png')
plt.close()
# plt.show()
