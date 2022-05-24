import os
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from seq2contact import *
from sklearn.linear_model import LinearRegression


def repair_dataset_11_14_21():
    dir = '/project2/depablo/erschultz/dataset_11_14_21/samples'
    # dir = '/home/eric/sequences_to_contact_maps/dataset_11_14_21/samples'
    chi = np.array([[-1,1.8,-0.5,1.8,0.1,1.3,-0.1,0.1,0.8,1.4,2,1.7,1.5,-0.2,1.1],
                    [0,-1,-0.6,0.6,0.8,-0.8,-0.7,-0.1,0,-0.4,-0.2,0.6,-0.9,1.4,0.3],
                    [0,0,-1,1.6,0,-0.2,-0.4,1.5,0.7,1.8,-0.7,-0.9,0.6,1,0.5],
                    [0,0,0,-1,0.8,1.3,-0.6,0.7,0.1,1.4,0.6,0.7,-0.6,0.5,0.5],
                    [0,0,0,0,-1,0.9,0.2,1.5,1.7,0.1,-0.7,0.8,0.7,1.6,1.6],
                    [0,0,0,0,0,-1,0.6,-0.2,0.8,0.7,-1,-0.9,1.6,0.8,0.3],
                    [0,0,0,0,0,0,-1,-0.2,-0.6,1.8,-0.6,1.9,1.1,0.4,-0.4],
                    [0,0,0,0,0,0,0,-1,1.7,-0.4,1.7,0.2,1.2,1.8,-0.1],
                    [0,0,0,0,0,0,0,0,-1,0.7,0.2,0.8,-0.4,1.4,1.3],
                    [0,0,0,0,0,0,0,0,0,-1,-0.4,0.5,1.9,0.1,0.1],
                    [0,0,0,0,0,0,0,0,0,0,-1,0.9,1,1.3,1],
                    [0,0,0,0,0,0,0,0,0,0,0,-1,1.5,-0.1,0.7],
                    [0,0,0,0,0,0,0,0,0,0,0,0,-1,0.6,-0.6],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0.2],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1]])

    for i in range(1, 2001):
        sample_dir = osp.join(dir, f'sample{i}')
        if not osp.exists(sample_dir):
            print("{sample_dir} doesn't exist")
            continue
        x = np.load(osp.join(sample_dir, 'x.npy'))
        psi = np.zeros((1024, 15)) # transformation of x such that S = psi \chi psi^T
        psi[:, 0] = (np.sum(x[:, 0:3], axis = 1) == 1) # exactly 1 of A, B, C
        psi[:, 1] = (np.sum(x[:, 0:3], axis = 1) == 2) # exactly 2 of A, B, C
        psi[:, 2] = (np.sum(x[:, 0:3], axis = 1) == 3) # A, B, and C
        psi[:, 3] = x[:, 3] # D
        psi[:, 4] = x[:, 4] # E
        psi[:, 5] = np.logical_and(x[:, 3], x[:, 4]) # D and E
        psi[:, 6] = np.logical_and(x[:, 3], x[:, 5]) # D and F
        psi[:, 7] = np.logical_xor(x[:, 0], x[:, 5]) # either A or F
        psi[:, 8] = x[:, 6] # G
        psi[:, 9] = np.logical_and(np.logical_and(x[:, 6], x[:, 7]), np.logical_not(x[:, 4])) # G and H and not E
        psi[:, 10] = x[:, 7] # H
        psi[:, 11] = x[:, 8] # I
        psi[:, 12] = x[:, 9] # J
        psi[:, 13] = np.logical_or(x[:, 7], x[:, 8]) # H or I
        psi[:, 14] = np.logical_xor(x[:, 8], x[:, 9]) # either I or J
        np.save(osp.join(sample_dir, 'psi.npy'), psi)

def check_seq(dataset):
    dir = osp.join("/project2/depablo/erschultz", dataset, "samples")
    dir = osp.join("/home/erschultz", dataset, "samples")
    ids = set()
    for file in os.listdir(dir):
        if file.startswith('sample'):
            id = int(file[6:])
            file_dir = osp.join(dir, file)
            try:
                x, psi, chi, e, s, y, ydiag = load_all(path, data_folder = dataFolder,
                                                    save = False,
                                                    throw_exception = True)

                m, k = psi.shape
                seq = np.zeros((m, k))
                for i in range(k):
                    seq_i = np.loadtxt(osp.join(file_dir, 'seq{}.txt'.format(i)))
                    seq[:, i] = seq_i

                if not np.array_equal(seq, psi):
                    print(psi)
                    print(seq)
                    print(id)
                    ids.add(id)
            except Exception as e:
                print(e, f'id={id}')
                ids.add(id)
                continue

    print(ids, len(ids))

def makeDirsForMaxEnt(dataset):
    for sample in [1]:
        dir = '/home/erschultz/sequences_to_contact_maps'
        sample_folder = osp.join(dir, dataset, 'samples', f'sample{sample}')
        assert osp.exists(sample_folder)

        for method in ['PCA', 'PCA-normalize']:
            os.mkdir(osp.join(sample_folder, method), mode = 0o755)
            for k in [2, 4, 6, 8]:
                os.mkdir(osp.join(sample_folder, method, f'k{k}'), mode = 0o755)
                for replicate in [1]:
                    os.mkdir(osp.join(sample_folder, method, f'k{k}', f'replicate{replicate}'), mode = 0o755)

def test_robust_PCA():

    if False:
        dir = '/home/eric/dataset_test/rpca_test'
        n = 1000
        r = 10
        x = np.random.rand(n, r)
        y = np.random.rand(r, n)
        l_0 = x @ y
        s_0 = np.random.binomial(n = 1, p = 0.3, size = (n, n))

        inp = l_0 + s_0

        plotContactMap(l_0, ofile = osp.join(dir, 'L.png'), vmax = 'max')
        plotContactMap(s_0, ofile = osp.join(dir, 'S.png'), vmax = 1)
        plotContactMap(inp, ofile = osp.join(dir, 'inp.png'), vmax = 'max')

        L, S = R_pca(inp).fit(max_iter=200)
        plotContactMap(L, ofile = osp.join(dir, 'RPCA_L.png'), vmax = np.mean(y))
        plotContactMap(S, ofile = osp.join(dir, 'RPCA_S.png'), vmax = np.mean(y))

    if True:
        dir = '/home/eric/dataset_test/rpca_test2'
        dataset_test = '/home/eric/dataset_test/samples'
        l0 = np.load(osp.join(dataset_test, 'sample20/PCA_analysis/y_diag_rank_1.npy'))
        p = np.load(osp.join(dataset_test, 'sample22/y.npy'))
        p = p / np.max(p) + 1e-8
        # m = np.load(osp.join(dataset_test, 'sample21/y.npy'))
        m = l0*p
        s0 = m - l0
        l0_log = np.log(l0)
        p_log = np.log(p)
        print('p', np.min(p_log))
        m_log = np.log(m)
        print(np.min(m_log))
        # plotContactMap(l0, ofile = osp.join(dir, 'L0.png'), vmax = 'max')
        # plotContactMap(l0_log, ofile = osp.join(dir, 'L0_log.png'), vmax = 'max')
        # plotContactMap(m, ofile = osp.join(dir, 'M.png'), vmax = 'mean')
        # plotContactMap(m_log, ofile = osp.join(dir, 'M_log.png'), vmin = 'min', vmax = 'max')
        # plotContactMap(s0, ofile = osp.join(dir, 'S0.png'), vmin = 'min', vmax = 'mean', cmap='blue-red')

        # plotContactMap(p, ofile = osp.join(dir, 'P.png'), vmax = 'mean')
        # plotContactMap(p_log, ofile = osp.join(dir, 'P_log.png'), vmin = 'min', vmax = 'max')

        # L, S = R_pca(m).fit(max_iter=200)
        # plotContactMap(L, ofile = osp.join(dir, 'RPCA_L.png'), vmax = 'mean')
        # plotContactMap(S, ofile = osp.join(dir, 'RPCA_S.png'), vmin = 'min', vmax = 'max', cmap='blue-red')
        L_log, S_log = R_pca(m_log).fit(max_iter=2000)
        plotContactMap(L_log, ofile = osp.join(dir, 'RPCA_L_log.png'), vmin = 'min', vmax = 'max')
        plotContactMap(S_log, ofile = osp.join(dir, 'RPCA_S_log.png'), vmin = 'min', vmax = 'max', cmap='blue-red')
        L_log_exp = np.exp(L_log)
        S_log_exp = np.exp(S_log)
        plotContactMap(L_log_exp, ofile = osp.join(dir, 'RPCA_L_log_exp.png'), vmax = 'max')
        plotContactMap(S_log_exp, ofile = osp.join(dir, 'RPCA_S_log_exp.png'), vmin = 'min', vmax = 'max', cmap='blue-red')

        PC_m = plot_top_PCs(m, inp_type='m', verbose = True, odir = dir, plot = True)
        meanDist = genomic_distance_statistics(m)
        m_diag = diagonal_preprocessing(m, meanDist)
        PC_m_diag = plot_top_PCs(m_diag, inp_type='m_diag', verbose = True, odir = dir, plot = True)

        # plot_top_PCs(L_log_exp, inp_type='L_log_exp', verbose = True, odir = dir, plot = True)
        PC_L_log = plot_top_PCs(L_log, inp_type='L_log', verbose = True, odir = dir, plot = True)
        stat = pearsonround(PC_L_log[0], PC_m[0])
        print("Correlation between PC 1 of L_log and M: ", stat)
        stat = pearsonround(PC_L_log[1], PC_m[1])
        print("Correlation between PC 2 of L_log and M: ", stat)
        meanDist = genomic_distance_statistics(L_log)
        L_log_diag = diagonal_preprocessing(L_log, meanDist)
        PC_L_log_diag = plot_top_PCs(L_log_diag, inp_type='L_log_diag', verbose = True, odir = dir, plot = True)
        stat = pearsonround(PC_L_log_diag[0], PC_m_diag[0])
        print("Correlation between PC 1 of L_log_diag and M_diag: ", stat)
        stat = pearsonround(PC_L_log_diag[1], PC_m_diag[1])
        print("Correlation between PC 2 of L_log_diag and M_diag: ", stat)
        # plot_top_PCs(m_log, inp_type='m_log', verbose = True, odir = dir, plot = True)
        # meanDist = genomic_distance_statistics(m_log)
        # m_log_diag = diagonal_preprocessing(m_log, meanDist)
        # plot_top_PCs(m_log_diag, inp_type='m_log_diag', verbose = True, odir = dir, plot = True)


        # ydiag = np.load(osp.join(dataset_test, 'sample21/y_diag.npy'))
        # ydiag_rank_1 = np.load(osp.join(dataset_test, 'sample21/PCA_analysis/y_diag_rank_1.npy'))
        # dif = ydiag - ydiag_rank_1
        # plotContactMap(ydiag, ofile = osp.join(dir, 'ydiag.png'), vmax = 'max')
        # plotContactMap(ydiag_rank_1, ofile = osp.join(dir, 'ydiag_rank_1.png'), vmax = 'max')
        # plotContactMap(dif, ofile = osp.join(dir, 'dif.png'), vmin = 'min', vmax = 'max', cmap='blue-red')


    if False:
        # dir = '/home/eric/dataset_test/samples/sample104'
        dir = '/home/eric/sequences_to_contact_maps/dataset_09_21_21/samples/sample1'
        y = np.load(osp.join(dir, 'y.npy'))
        # L, S = R_pca(y).fit(max_iter=200)
        # plotContactMap(L, ofile = osp.join(dir, 'RPCA_L.png'), vmax = np.mean(y))
        # plotContactMap(S, ofile = osp.join(dir, 'RPCA_S.png'), vmax = np.mean(y))

        # l = 1/np.sqrt(1024) * 1/100
        y_diag = np.load(osp.join(dir, 'y_diag.npy'))
        L, S = R_pca(y_diag).fit(max_iter=200)
        plotContactMap(L, ofile = osp.join(dir, 'RPCA_L_diag.png'), vmin='min', vmax = 'max')
        plotContactMap(S, ofile = osp.join(dir, 'RPCA_S_diag.png'), vmin='min', vmax = np.max(S), cmap='blue-red')
        plotContactMap(y_diag, ofile = osp.join(dir, 'y_diag.png'), vmax = 'max')
        plot_top_PCs(L, verbose = True)

def scc_y_vs_y_rank1():
    dir = '/home/eric/sequences_to_contact_maps/dataset_01_16_22/samples/sample40'
    y = np.load(osp.join(dir, 'y.npy'))
    ycopy = y.copy()
    y_diag = np.load(osp.join(dir, 'y_diag.npy'))

    pca = PCA(n_components = 1)
    y_transform = pca.fit_transform(y)
    yhat = pca.inverse_transform(y_transform)
    y_transform = pca.fit_transform(y_diag)
    yhat_diag = pca.inverse_transform(y_transform)
    plotContactMap(yhat, ofile = osp.join(dir, 'yhat_rank1.png'), vmax = 'max')
    plotContactMap(yhat_diag, ofile = osp.join(dir, 'yhat_diag_rank1.png'), vmax = 'max')

    pca = PCA(n_components = 10)
    pca.fit(ycopy)
    plot_top_PCs(ycopy, 'name', dir)


    m, _ = y.shape
    triu_ind = np.triu_indices(m)

    overall_corr, _ = pearsonr(y[triu_ind], yhat[triu_ind])
    print('overall', overall_corr)
    scc_A, _ = pearsonr(y_diag[triu_ind], yhat_diag[triu_ind])
    print('scc_A', scc_A)


    corr_arr = np.zeros(m-2)
    corr_arr[0] = np.NaN
    scc_B = 0
    denom = np.sum([len(np.diagonal(y, offset = d)) for d in list(range(1, m-2))])
    weights = 0
    for d in range(1, m-2):
        # n-1, n, and 0 are NaN always, so skip
        y_diag = np.diagonal(y, offset = d)
        yhat_diag = np.diagonal(yhat, offset = d)
        corr, _ = pearsonr(y_diag, yhat_diag)
        corr_arr[d] = corr
        weight = len(y_diag) / denom
        weights += weight
        scc_B += weight * corr
    print(corr_arr)
    print('scc_B', scc_B)

    avg = np.nanmean(corr_arr)
    print('avg', avg)

    plt.plot(np.arange(m-2), corr_arr, color = 'black')
    plt.ylim(-0.5, 1)
    plt.xlabel('Distance', fontsize = 16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize = 16)

    plt.tight_layout()
    plt.savefig(osp.join(dir, 'distance_pearson_y_vs_y_rank_1.png'))
    plt.close()

def is_scc_weighted_mean():
    dir = '/home/eric/sequences_to_contact_maps/dataset_01_15_22/samples/sample40'
    replicate = '/home/eric/sequences_to_contact_maps/dataset_01_15_22/samples/sample40/PCA/k4/replicate1/iteration101'
    y = np.load(osp.join(dir, 'y.npy'))
    y_diag = np.load(osp.join(dir, 'y_diag.npy'))
    yhat = np.load(osp.joiPCAn(replicate, 'y.npy'))
    yhat_diag = np.load(osp.join(replicate, 'y_diag.npy'))

    m, _ = y.shape
    triu_ind = np.triu_indices(m)

    overall_corr, _ = pearsonr(y[triu_ind], yhat[triu_ind])
    print(overall_corr)
    scc_A, _ = pearsonr(y_diag[triu_ind], yhat_diag[triu_ind])
    print(scc_A)


    corr_arr = np.zeros(m-2)
    corr_arr[0] = np.NaN
    scc_B = 0
    denom = np.sum([d for d in list(range(1, m-2))])
    for d in range(1, m-2):
        # n-1, n, and 0 are NaN always, so skip
        y_diag = np.diagonal(y, offset = d)
        yhat_diag = np.diagonal(yhat, offset = d)
        corr, _ = pearsonr(y_diag, yhat_diag)
        corr_arr[d] = corr
        weight = d / denom
        scc_B += weight * corr
    print(corr_arr)
    print(scc_B)

    avg = np.nanmean(corr_arr)
    print(avg)





if __name__ == '__main__':
    # repair_dataset_11_14_21()
    # is_scc_weighted_mean()
    # scc_y_vs_y_rank1()
    # test_robust_PCA()
    # check_seq()
    # find_mising_ids()
    # makeDirsForMaxEnt("dataset_04_27_22")
