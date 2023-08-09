import sys
import os.path as osp
import numpy as np
import scipy
import matplotlib.pyplot as plt

import seaborn as sns
from pylib.utils.energy_utils import (calculate_all_energy, calculate_D,
                                      calculate_diag_chi_step, calculate_L,
                                      calculate_S)

sys.path.append('/home/erschultz/TICG-chromatin/scripts')
from makeLatexTable_new import *

sys.path.append('/home/erschultz/sequences_to_contact_maps')
from scripts.load_utils import (get_final_max_ent_folder,
                                load_import_log, load_L)
from scripts.utils import DiagonalPreprocessing, pearson_round
from scripts.xyz_utils import xyz_load, xyz_write
from scripts.plotting_utils import RED_CMAP

test=False
label_fontsize=24
tick_fontsize=22
letter_fontsize=26
dataset = 'dataset_02_04_23'; sample = 208; GNN_ID = 434
# dataset = 'dataset_04_05_23'; sample = 1001; GN_ID = 407
# dataset = 'dataset_04_05_23'; sample = 1001; GNN_ID = 423
samples_list = [208, 209, 210, 211, 212, 213, 214, 215, 223, 224]

k=10
def get_dirs(sample_dir):
    grid_dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03')
    max_ent_dir = f'{grid_dir}-max_ent{k}'
    gnn_dir = f'{grid_dir}-GNN{GNN_ID}'

    return max_ent_dir, gnn_dir


def get_y(sample_dir):
    max_ent_dir, gnn_dir = get_dirs(sample_dir)
    y = np.load(osp.join(sample_dir, 'y.npy')).astype(np.float64)
    y /= np.mean(np.diagonal(y))

    final = get_final_max_ent_folder(max_ent_dir)
    y_pca = np.load(osp.join(final, 'y.npy')).astype(np.float64)
    y_pca /= np.mean(np.diagonal(y_pca))

    y_gnn = np.load(osp.join(gnn_dir, 'y.npy')).astype(np.float64)
    y_gnn /= np.mean(np.diagonal(y_gnn))

    return y, y_pca, y_gnn


sample_dir = f'/home/erschultz/{dataset}/samples/sample{sample}'
max_ent_dir, gnn_dir = get_dirs(sample_dir)
y, y_pca, y_gnn = get_y(sample_dir)
meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
y_diag = DiagonalPreprocessing.process(y, meanDist)
m = len(y)

result = load_import_log(sample_dir)
start = result['start_mb']
end = result['end_mb']
chrom = result['chrom']
resolution = result['resolution']
resolution_mb = result['resolution_mb']
all_labels = np.linspace(start, end, len(y))
all_labels = np.round(all_labels, 0).astype(int)
# genome_ticks = [0, len(y)//3, 2*len(y)//3, len(y)-1]
genome_ticks = [0, len(y)-1]
genome_labels = [f'{all_labels[i]}' for i in genome_ticks]

final = get_final_max_ent_folder(max_ent_dir)
with open(osp.join(final, 'distance_pearson.json'), 'r') as f:
    # TODO this is after final iteration, not convergence
    pca_results = json.load(f)
    pca_scc = np.round(pca_results["scc_var"], 3)
meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_pca)
y_pca_diag = DiagonalPreprocessing.process(y_pca, meanDist)
L = load_L(max_ent_dir)
with open(osp.join(final, 'config.json'), 'r') as f:
    config = json.load(f)
diag_chis_continuous = calculate_diag_chi_step(config)
D = calculate_D(diag_chis_continuous)
S = calculate_S(L, D)


with open(osp.join(gnn_dir, 'distance_pearson.json'), 'r') as f:
    gnn_results = json.load(f)
    gnn_scc = np.round(gnn_results["scc_var"], 3)


S_gnn = np.load(osp.join(gnn_dir, 'S.npy'))

def get_pcs(input, smooth=False, h=1, verbose=False):
    if input is None:
        return None

    if smooth:
        input = scipy.ndimage.gaussian_filter(input, (h, h))

    input = epilib.get_oe(input)
    # input = np.corrcoef(input)

    if verbose:
        print(input)
    seqs = epilib.get_pcs(input, 12,
                        normalize = False, align = True)
    return seqs.T # k x m

smooth = True; h = 1
pcs = get_pcs(y, smooth, h)
pcs_pca = get_pcs(y_pca, smooth, h)
pcs_gnn = get_pcs(y_gnn, smooth, h)
# pcs = epilib.get_pcs(epilib.get_oe(y), 12).T
# pcs_pca = epilib.get_pcs(epilib.get_oe(y_pca), 12).T
# pcs_gnn = epilib.get_pcs(epilib.get_oe(y_gnn), 12).T


# make sure they are aligned by ensuring corr is positive
pcs_pca[0] *= np.sign(pearson_round(pcs[0], pcs_pca[0]))
pcs_gnn[0] *= np.sign(pearson_round(pcs[0], pcs_gnn[0]))

# boxplot data
if not test:
    args = getArgs(data_folder = f'/home/erschultz/{dataset}',
                    samples = samples_list)
    args.experimental = True
    args.convergence_definition = None
    args.gnn_id = GNN_ID
    data, _ = load_data(args)
    max_ent = osp.split(max_ent_dir)[1]

    gnn = f'optimize_grid_b_140_phi_0.03-GNN{GNN_ID}'
    gnn_times = data[0][gnn]['total_time']
    gnn_sccs = data[0][gnn]['scc_var']
    gnn_pearsons = data[0][gnn]['pearson_pc_1']

    max_ent_times_strict = data[k][max_ent]['converged_time']
    max_ent_times_strict = [i for i in max_ent_times_strict if i is not None]
    max_ent_sccs_strict = data[k][max_ent]['scc_var']
    max_ent_sccs_strict = [i for i in max_ent_sccs_strict if not np.isnan(i)]
    max_ent_pearsons_strict = data[k][max_ent]['pearson_pc_1']

    max_ent += '_stop'
    max_ent_times = data[k][max_ent]['converged_time']
    max_ent_times = [i for i in max_ent_times if i is not None]
    max_ent_sccs = data[k][max_ent]['scc_var']
    max_ent_sccs = [i for i in max_ent_sccs if not np.isnan(i)]
    max_ent_pearsons = data[k][max_ent]['pearson_pc_1']
else:
    max_ent_times = np.random.normal(size=100)
    max_ent_sccs = np.random.normal(loc=0.7, scale = 0.1, size=100)
    max_ent_times_strict = np.random.normal(size=100)
    max_ent_sccs_strict = np.random.normal(loc=0.75, scale = 0.05, size=100)
    max_ent_pearsons = np.random.normal(size=100)
    max_ent_pearsons_strict = np.random.normal(size=100)
    gnn_times = np.random.normal(size=100)
    gnn_sccs = np.random.normal(loc=0.6, scale = 0.12, size=100)
    gnn_pearsons = np.random.normal(size=100)


# xyz
file = osp.join(gnn_dir, 'production_out/output.xyz')
xyz = xyz_load(file, multiple_timesteps=True)[::, :m, :]

# if not test:
#     y_chr = np.load(osp.join('/home/erschultz', dataset, f'chroms_50k/sample{chrom}/y.npy'))
#     y_chr_diag = epilib.get_oe(y_chr)
#     seq = epilib.get_pcs(y_chr_diag, 1, normalize = True)[:, 0]
#     x = np.zeros((len(seq), 2))
#     # kmeans = KMeans(n_clusters = 2)
#     # kmeans.fit(y_chr_diag)
#
#     # x[np.arange(len(y_chr)), kmeans.labels_] = 1
#
#     # ensure that positive PC is compartment A
#     chip_seq = np.load(f'/media/erschultz/1814ae69-5346-45a6-b219-f77f6739171c/home/erschultz/chip_seq_data/GM12878/hg19/signal_p_value/ENCFF850KGH/{chrom}.npy')
#     chip_seq = chip_seq[:, 1]
#     corr = pearson_round(chip_seq, seq, stat = 'spearman')
#     seq *= np.sign(corr)
#     seq = np.sign(seq)
#
#     print(start / resolution_mb, end / resolution_mb)
#     left = int(start / resolution_mb)
#     right = int(end / resolution_mb)
#     x = x[left:right]

x = np.zeros((m, 2))
y_diag = epilib.get_oe(y)
print(y.shape, y_diag.shape)
seq = epilib.get_pcs(y_diag, 1, normalize = True)[:, 0]
seq_a = np.zeros_like(seq)
seq_b = np.zeros_like(seq)
seq_a[seq > 0] = 1
seq_b[seq < 0] = 1

# ensure that positive PC is compartment A based on count of A-A contacts
# a compartment should have fewer self-self contacts as it is less dense
count_a = seq_a @ y @ seq_a
count_b = seq_b @ y @ seq_b
print('self-self counts: ', count_a, count_b)
if count_a < count_b:
    x[:, 0] = seq_a
    x[:, 1] = seq_b
else:
    x[:, 0] = seq_b
    x[:, 1] = seq_a

xyz_write(xyz, osp.join(gnn_dir, 'xyz.xyz'), 'w', x = x)


def figure(test=False):
    ### combined figure ###
    print('---'*9)
    print('Starting Figure')
    fig = plt.figure(figsize=(18, 10.5))
    ax1 = plt.subplot(2, 24, (1, 6))
    ax2 = plt.subplot(2, 24, (9, 14))
    # ax_cb = plt.subplot(2, 48*2, 29*2-1)
    ax4 = plt.subplot(2, 24, (17, 24)) # pc
    # ax4_2 = plt.subplot(4, 24, (41, 48)) # diagonal
    ax5 = plt.subplot(2, 24, (25, 29)) # meandist
    ax6 = plt.subplot(2, 48, (63, 67))
    ax7 = plt.subplot(2, 48, (72, 76))
    ax8 = plt.subplot(2, 48, (81, 85))
    axes = [ax1, ax4, ax5, ax6, ax7, ax8]


    vmin = 0
    vmax = np.mean(y)
    # combined hic maps
    npixels = np.shape(y)[0]
    indu = np.triu_indices(npixels)
    indl = np.tril_indices(npixels)
    data = zip([y_gnn, y_pca], ['GNN', 'Max Ent'], [ax1, ax2], [gnn_scc, pca_scc])
    for i, (y_sim, label, ax, scc) in enumerate(data):
        # make composite contact map
        composite = np.zeros((npixels, npixels))
        composite[indu] = y_sim[indu]
        composite[indl] = y[indl]

        s = sns.heatmap(composite, linewidth = 0, vmin = vmin, vmax = vmax, cmap = RED_CMAP,
                        ax = ax, cbar = None)
        if i == 0:
            s.set_yticks(genome_ticks, labels = genome_labels,
                    fontsize = tick_fontsize)
        else:
            s.set_yticks([])
        title = (f'{label} ' + r'$\hat{H}$')
        print(title, f': SCC={scc}')
        # s.set_title(title, fontsize = label_fontsize)
        s.axline((0,0), slope=1, color = 'k', lw=1)
        s.text(0.99*m, 0.01*m, label, fontsize=letter_fontsize, ha='right', va='top', weight='bold')
        s.text(0.01*m, 0.99*m, 'Experiment', fontsize=letter_fontsize, weight='bold')
        s.set_xticks(genome_ticks, labels = genome_labels, rotation = 0,
                    fontsize = tick_fontsize)

    ax4.plot(pcs[0], label = 'Experiment', color = 'k')
    ax4.plot(pcs_pca[0], label = f'Max Ent', color = 'b')
    ax4.plot(pcs_gnn[0], label = f'GNN', color = 'r')
    print(f'Max Ent (r={pearson_round(pcs[0], pcs_pca[0])})', '\n',
            f'GNN (r={pearson_round(pcs[0], pcs_gnn[0])})')
    ax4.set_xticks(genome_ticks, labels = genome_labels, rotation = 0,
                    fontsize = tick_fontsize)
    ax4.set_yticks([])
    ax4.set_ylabel('PC 1', fontsize=label_fontsize)
    # ax4.set_title(f'PC 1\nCorr(Exp, GNN)={pearson_round(pcs[0], pcs_gnn[0])}')
    ax4.set_ylim(None, .14)
    ax4.legend(bbox_to_anchor=(0.5, .98), loc="upper center",
                fontsize = 14, borderaxespad=0, ncol=3)

    # resized = rotate_bound(y,-45)
    # height=50
    # center = resized.shape[0] // 2
    # resized = resized[center-height:center, :]
    # sns.heatmap(resized, ax = ax4_2, linewidth = 0, vmin = vmin, vmax = vmax,
    #             cmap = RED_CMAP, cbar = False)
    # ax4_2.set_xticks([])
    # ax4_2.set_yticks([])


    # compare P(s)
    meanDist = DiagonalPreprocessing.genomic_distance_statistics(y, 'prob')
    meanDist_pca = DiagonalPreprocessing.genomic_distance_statistics(y_pca, 'prob')
    meanDist_gnn = DiagonalPreprocessing.genomic_distance_statistics(y_gnn, 'prob')
    log_labels = np.linspace(0, resolution*(len(meanDist)-1), len(meanDist))
    data = zip([meanDist, meanDist_pca, meanDist_gnn],
                ['Experiment', 'Max Ent', f'GNN'],
                ['k', 'b', 'r'])
    for arr, fig_label, c in data:
        print(fig_label, mean_squared_error(meanDist, arr))
        ax5.plot(log_labels, arr, label = fig_label, color = c)

    ax5.set_yscale('log')
    ax5.set_xscale('log')
    ax5.set_ylabel('Contact Probability', fontsize = label_fontsize)
    ax5.set_xlabel('Genomic Separation (bp)', fontsize = label_fontsize)
    ax4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # ax5.legend(loc='upper right', fontsize=legend_fontsize)

    # time and scc
    labels = ['Max Ent', 'Max Ent (long)', 'GNN']
    ticks = range(1, 4)
    data = [max_ent_sccs, max_ent_sccs_strict, gnn_sccs]
    print(max_ent_sccs, np.mean(max_ent_sccs))
    print(gnn_sccs, np.mean(gnn_sccs))
    b1 = ax6.boxplot(data, vert = True,
                        patch_artist = True, labels = labels)
    ax6.set_ylim()
    ax6.set_ylabel(r'SCC(H$^{\rm sim}$, H$^{\rm exp}$)', fontsize=label_fontsize)



    data = [max_ent_pearsons_strict, max_ent_pearsons, gnn_pearsons]
    print(max_ent_pearsons_strict)
    b3 = ax7.boxplot(data, vert = True,
                        patch_artist = True, labels = labels)
    # axes[1].set_yscale('log')
    ax7.set_ylabel(r'Pearson(PC1$^{\rm sim}$, PC1$^{\rm exp}$)', fontsize=label_fontsize)

    data = [max_ent_times, max_ent_times_strict, gnn_times]
    b2 = ax8.boxplot(data,  vert = True,
                        patch_artist = True, labels = labels)
    # ax8.set_yticks([10, 50, 100])
    # ax8.set_yscale('log')
    ax8.set_ylim(0, 90)
    ax8.set_ylabel('Time (mins)', fontsize=label_fontsize)

    # fill with colors
    colors = ['b', 'b', 'r']
    for bplot in [b1, b2, b3]:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set(edgecolor='black', linewidth=2)

    # rotate xticks
    # Create offset transform by 5 points in x direction
    dx = 15/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for ax in [ax6, ax7, ax8]:
        ax.set_xticks(ticks, labels, rotation=40, ha='right')
        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # axes.append(ax2)
    for n, ax in enumerate(axes):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    axes[-1].text(1.35, 1.05, string.ascii_uppercase[n+1], transform=ax.transAxes,
            size=letter_fontsize, weight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17, top = 0.95, left = 0.1, right = 0.95,
                    hspace = 0.25)
    # if test:
    #     plt.show()
    # else:
    plt.savefig('/home/erschultz/TICG-chromatin/figures/figure2.png')
    plt.close()

def supp_figure():
    ### combined figure ###
    print('---'*9)
    print('Starting Figure')
    fig, axes = plt.subfigures(1, 3)
    fig.set_figheight(5.5)
    fig.set_figwidth(18)
    ax1, ax2, ax3 = axes

    # time and scc
    labels = ['Max Ent', 'Max Ent (long)', 'GNN']
    ticks = range(1, 4)
    data = [max_ent_sccs, max_ent_sccs_strict, gnn_sccs]
    print(max_ent_sccs, np.mean(max_ent_sccs))
    print(gnn_sccs, np.mean(gnn_sccs))
    b1 = ax1.boxplot(data, vert = True,
                        patch_artist = True, labels = labels)
    ax1.set_ylim()
    ax1.set_ylabel(r'SCC(H$^{\rm sim}$, H$^{\rm exp}$)', fontsize=label_fontsize)


    data = [max_ent_pearsons_strict, max_ent_pearsons, gnn_pearsons]
    print(max_ent_pearsons_strict)
    b3 = ax2.boxplot(data, vert = True,
                        patch_artist = True, labels = labels)
    # axes[1].set_yscale('log')
    ax2.set_ylabel(r'Pearson(PC1$^{\rm sim}$, PC1$^{\rm exp}$)', fontsize=label_fontsize)

    data = [max_ent_times, max_ent_times_strict, gnn_times]
    b2 = ax7.boxplot(data,  vert = True,
                        patch_artist = True, labels = labels)
    # ax8.set_yticks([10, 50, 100])
    # ax8.set_yscale('log')
    ax7.set_ylim(0, 90)
    ax7.set_ylabel('Time (mins)', fontsize=label_fontsize)

    # fill with colors
    colors = ['b', 'b', 'r']
    for bplot in [b1, b2, b3]:
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set(edgecolor='black', linewidth=2)

    # rotate xticks
    # Create offset transform by 5 points in x direction
    dx = 15/72.; dy = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for ax in axes:
        ax.set_xticks(ticks, labels, rotation=40, ha='right')
        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    for n, ax in enumerate(axes):
        ax.text(-0.1, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
                size=letter_fontsize, weight='bold')

    plt.tight_layout()
    plt.savefig('/home/erschultz/TICG-chromatin/figures/figure2_supp.png')
    plt.close()



if __name__ == '__main__':
    figure()
    supp_figure()
