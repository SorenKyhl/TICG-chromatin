import math

import numpy as np
import scipy


def calculate_all_energy(config, seq, chi, diag_chis=None):
    assert len(seq.shape) == 2
    if seq.shape[1] > seq.shape[0]:
        seq = seq.T
    diag_chis_continuous = calculate_diag_chi_step(config, diag_chis)
    D = calculate_D(diag_chis_continuous)
    L = calculate_L(seq, chi)
    S = calculate_S(L, D)
    return L, D, S

def calculate_S(L, D):
    # S is symmetric net energy
    if D is None:
        return L
    elif L is None:
        return D
    else:
        S = L + D
        return S

def convert_L_to_Lp(L):
    # Lp only requires upper triangle
    return L + L.T - np.diag(np.diagonal(L).copy())

def calculate_Lp(x, chi):
    L = calculate_L(x, chi)
    Lp = convert_L_to_Lp(L)
    return Lp

def calculate_L(psi, chi):
    if psi is None or chi is None:
        return None
    assert len(chi.shape) == 2, f"chi has shape {chi.shape}"
    if psi.shape[1] > psi.shape[0]:
        psi = psi.T
    # zero lower triangle (double check)
    chi = np.triu(chi)

    try:
        L = psi @ chi @ psi.T
    except ValueError as e:
        print('psi', psi, psi.shape)
        print('chi', chi, chi.shape)
        raise
    L = (L+L.T)/2 # make symmetric

    return L

def calculate_diag_chi_step(config, diag_chi = None):
    m = config['nbeads']
    if diag_chi is None:
        diag_chi = config['diag_chis']
    diag_bins = len(diag_chi)

    if diag_bins == m:
        return diag_chi

    if 'diag_start' in config.keys():
        diag_start = config['diag_start']
    else:
        diag_start = 0

    if 'diag_cutoff' in config.keys():
        diag_cutoff = config['diag_cutoff']
    else:
        diag_cutoff = m

    if 'dense_diagonal_on' in config.keys():
        dense = config['dense_diagonal_on']
    else:
        dense = False

    if dense:
        if 'n_small_bins' in config.keys():
            n_small_bins = config['n_small_bins']
            small_binsize = config['small_binsize']
            big_binsize = config['big_binsize']
        else:
            # soren compatibility
            n_small_bins = int(config['dense_diagonal_loading'] * diag_bins)
            n_big_bins = diag_bins - n_small_bins
            m_eff = diag_cutoff - diag_start # number of beads with nonzero interaction
            dividing_line = m_eff * config['dense_diagonal_cutoff']
            small_binsize = int(dividing_line / (n_small_bins))
            big_binsize = int((m_eff - dividing_line) / n_big_bins)

    diag_chi_step = np.zeros(m)
    for d in range(diag_cutoff):
        if d < diag_start:
            continue
        d_eff = d - diag_start
        if dense:
            dividing_line = n_small_bins * small_binsize

            if d_eff > dividing_line:
                bin = n_small_bins + math.floor( (d_eff - dividing_line) / big_binsize)
            else:
                bin =  math.floor( d_eff / small_binsize)
        else:
            binsize = m / diag_bins
            bin = int(d_eff / binsize)
        diag_chi_step[d] = diag_chi[bin]

    return diag_chi_step

def diag_chi_step_to_dense(diag_chi_step, n_small_bins, small_binsize,
                            n_big_bins, big_binsize):
    diag_chi = np.zeros(n_small_bins + n_big_bins)

    left = 0
    right = left + small_binsize
    for bin in range(n_small_bins):
        diag_chi[bin] = np.mean(diag_chi_step[left:right])
        left += small_binsize
        right += small_binsize

    right = left + big_binsize
    for bin in range(n_big_bins):
        diag_chi[bin+n_small_bins] = np.mean(diag_chi_step[left:right])
        left += big_binsize
        right += big_binsize

    return diag_chi


def calculate_D(diag_chi_continuous):
    return scipy.linalg.toeplitz(diag_chi_continuous)

def test():
    m = 100
    I = np.random.choice([0, 1], size=(m*m,)).reshape(m, m)
    I = np.triu(I) + np.triu(I, 1).T
    print(I)

    psi = np.random.rand(m, 5)
    print('psi', psi)

    chi = np.random.rand(5,5)*5
    chi = np.triu(chi)
    print('chi\n', chi)

    Lp = calculate_L_prime(psi, chi)
    L = calculate_L(psi, chi)

    d = np.linspace(0, 2, m)
    D = calculate_D(d)
    print('D\n', D)

    def psi_chi_energy():
        print('psi, chi')
        sum = 0
        for i in range(m):
            for j in range(m):
                sum += I[i,j]* (psi[i] @ chi @ psi[j])
        print(sum)

    def Lp_energy():
        print('Lp')
        sum = 0
        for i in range(m):
            for j in range(m):
                sum += I[i,j]*Lp[i,j]

        print(sum)

    def L_energy():
        print('L')
        sum = 0
        for i in range(m):
            for j in range(i+1):
                sum += I[i,j]*L[i,j]

        print(sum)

    psi_chi_energy()
    Lp_energy()
    L_energy()

def test2():
    diag_chi_step = np.arange(0, 10, 1)
    print(diag_chi_step)
    diag_chi = diag_chi_step_to_dense(diag_chi_step, 4, 1, 2, 3)
    print(diag_chi)

if __name__ == '__main__':
    # test()
    test2()
