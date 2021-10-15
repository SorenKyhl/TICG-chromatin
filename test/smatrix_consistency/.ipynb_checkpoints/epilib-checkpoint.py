import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import straw
from os import system
import os

#import palettable
#from palettable.colorbrewer.sequential import Reds_3

mycmap = mpl.colors.LinearSegmentedColormap.from_list('custom',
                                             [(0,    'white'),
                                              (0.3,  'white'),
                                              (1,    '#ff0000')], N=126)


#%config InlineBackend.figure_format='retina'
#plt.rcParams['figure.figsize'] = [8,6]
#plt.rcParams.update({'font.size':14})

def import_contactmap_straw(filename, chrom=2, start=0, end=60575000, resolution=50000):

    '''
    loads Hi-C contact map using straw https://github.com/aidenlab/straw/wiki
    uses Knight-Ruiz matrix balancing (s.t. contact map is symmetric)
    see here for common files https://www.aidenlab.org/data.html
    
    example input for knight-rubin (KR) normalized chromosome 2, 0-130Mbp at 100kbp resolution.
    raw_data = straw.straw("KR", filename, "2:0:129999999", "2:0:129999999", "BP", 100000)
    '''
    
    basepairs = ":".join((str(chrom), str(start), str(end)))
    raw_data = straw.straw("KR", filename, basepairs, basepairs, "BP", resolution)
    raw_data = np.array(raw_data)

    # raw data is a map between locus pairs and reads -- need to pivot into a matrix. 
    df = pd.DataFrame(raw_data.transpose(), columns=['locus1 [bp]', 'locus2 [bp]', 'reads'])
    pivoted = df.pivot_table(values='reads', index='locus1 [bp]', columns='locus2 [bp]')
    xticks = np.array(pivoted.columns) # in base pairs
    
    filled = pivoted.fillna(0)
    trans = filled.transpose()
    hic = np.array(filled + trans)
    
    for i, row in enumerate(hic):
        # row normalization:
        #hic[i] /= sum(row)
        
        # normalize s.t. main diagonal has probability=1
        hic[i] /= hic[i,i]
        
    return hic, xticks

def get_contactmap(filename, atoms=1023):
    '''
    load contact map from file, stored as matrix
    '''
    df = pd.read_csv(filename, sep=" ", header=None, skiprows=0)
    #df /= df.stack().mean()
    df = np.array(df)

    df = df[0:atoms, 0:atoms]
    #df /= df[0][0]
    df /= max(np.array(df).diagonal())

    #df = np.log(df)
    return df

def get_diagonal(contact, plot=False):
    '''
    calculate the probablity of contact as a function of genomic distance 
    '''
    rows, cols = contact.shape
    d = np.zeros(rows)
    for k in range(rows):
        d[k] = np.mean(contact.diagonal(k))

    if plot:
        plt.figure(figsize=(12,10))
        plt.semilogy(d)
    return d

def norm_diagonal(contact, diagonal):
    '''
    normalize contact map by the diagonal
    '''
    norm = np.zeros_like(contact)

    for i,d in enumerate(diagonal):
        indices = np.arange(len(norm.diagonal(i)))
        norm[indices, indices+i] = d

    norm = norm + norm.transpose()

    return contact/norm

def plot_contactmap(contact, vmaxp=0.2):
    plt.figure(figsize=(12,10))
    sns.heatmap(contact, cmap=mycmap, vmin = 0, vmax = contact.mean()+vmaxp*contact.std())
    
def get_sequences(hic, k, plot=False, dtype="int"):
    '''
    calculate polymer bead sequences using k principal components
    returns 2*k epigenetic sequences
    '''
    OEmap = get_OEmap(hic)
    U, S, VT = np.linalg.svd(np.corrcoef(OEmap), full_matrices=0)
    
    pcs = []
    for i in range(k):
        pcs.append(np.array([up(n) for n in VT[i]]))
        pcs.append(-np.array([down(n) for n in VT[i]]))
        
        if plot:
            plt.figure()
            plt.plot(pcs[-2])
            plt.plot(pcs[-1])
    
    assert(dtype=="int" or dtype=="float")
    
    beads_per_bin = 1 
    seqs = []
    for pc in pcs:
        seqs.append(seq_from_pc(pc, beads_per_bin, dtype))
    
    return seqs

def get_OEmap(hic):
    diagonal = get_diagonal(hic)
    rows, cols = np.shape(hic)

    OEmap = np.zeros_like(hic)

    for i in range(rows):
        for j in range(cols):
            norm = diagonal[abs(i-j)]
            if norm:
                OEmap[i, j] = hic[i,j]/ norm
    return OEmap

def map_0_1(signal):
    '''
    maps signal data to the interval [0,1]
    '''
    signal = signal.real
    return (signal - np.min(signal))/(np.max(signal) - np.min(signal))

def seq_from_pc(pc, beads_per_bin = 1, dtype="int"):
    np.random.seed(1) 
    
    seq = map_0_1(pc)
    sequence = []
    for element in seq:
        for i in range(beads_per_bin):
            if dtype == "int":
                if np.random.rand() < element:
                    sequence.append(1)
                else:
                    sequence.append(0)
            elif dtype == "float":
                sequence.append(element)
    return sequence

def up(f):
    if f<0:
        return 0
    else:
        return f

def down(f):
    if f>0:
        return 0
    else:
        return f
    
def get_goals(hic, seqs):
    n = len(seqs)
    goals = []
    for i in range(n):
        for j in range(i,n):
            goals.append(np.mean((np.outer(seqs[i],seqs[j])*hic).flatten()))
            
    return goals

def plot_smooth(data, n_steps=20, llabel="none", fmt='o', norm=True):
    time_series_df = pd.DataFrame(data)
    smooth_path    = time_series_df.rolling(n_steps).mean()
    path_deviation = 2 * time_series_df.rolling(n_steps).std()

    under_line     = (smooth_path-path_deviation)[0]
    over_line      = (smooth_path+path_deviation)[0]

    #Plotting:
    if norm:
        s = np.array(range(len(data)))/len(data)
    else:
        s = np.array(range(len(data)))
    plt.semilogy(s, smooth_path, fmt, linewidth=2, label=llabel) #mean curve.
    plt.fill_between(s, under_line, over_line, color='b', alpha=.1) #std curves


def half_sample(contact):
    rows, cols = contact.shape
    i_vals = np.arange(0,int(rows/2),2)
    j_vals = np.arange(0,int(cols/2),2)

    x = np.zeros((int(rows/2), int(cols/2)))

    for i in range(int(rows/2)):
        for j in range(int(cols/2)):
            x[i][j] = 0.25*(contact[i*2][j*2] + contact[i*2+1][j*2] + contact[i*2][j*2+1] + contact[i*2+1][j*2+1])

    return x


def process(folder, ignore=100):
    data_out_filename = "data_out"
    contact = get_contactmap("./" + folder + "/" + data_out_filename + "/contacts.txt")
    #seq1 = np.array(pd.read_csv(folder + "/seq1.txt"))
    #seq2 = np.array(pd.read_csv(folder + "/seq2.txt"))
    seq1 = np.array(pd.read_csv(folder + "/pc1.txt"))
    seq2 = np.array(pd.read_csv(folder + "/pc2.txt"))

    goal_exp = np.zeros((2,2))
    goal_exp[0,0] = np.mean((np.outer(seq1,seq1)*contact).flatten())
    goal_exp[1,0] = np.mean((np.outer(seq1,seq2)*contact).flatten())
    goal_exp[1,1] = np.mean((np.outer(seq2,seq2)*contact).flatten())

    obs = pd.read_csv("./" + folder + "/" + data_out_filename + "/observables.traj", sep='\t', header=None)
    goal_sim = np.zeros((2,2))

    
    goal_sim[0,0] = np.mean(obs[1][ignore:])
    goal_sim[1,0] = np.mean(obs[2][ignore:])
    goal_sim[1,1] = np.mean(obs[3][ignore:])

    print("goals from experiment")
    print("calculated as weighted average from contact map")
    print(goal_exp)
    print("goals from simulation")
    print("calculated as mean of observables")
    print(goal_sim)
    
def process2(folder, ignore=100):
    data_out_filename = "production_out"
    contact = get_contactmap("./" + folder + "/" + data_out_filename + "/contacts.txt")
    #seq1 = np.array(pd.read_csv(folder + "/seq1.txt"))
    #seq2 = np.array(pd.read_csv(folder + "/seq2.txt"))
    
    seqs = []
    filenames = os.popen("ls "+folder+"../resources/pc*").read().split("\n")[:-1]
    print(filenames)
    for file in filenames:
        seqs.append( np.array(pd.read_csv(folder + "/" + file)) )
        
    print(seqs)
        
    k = len(seqs)
    goal_exp = np.zeros((k,k))
    for i, seqi in enumerate(seqs):
        for j, seqj in enumerate(seqs):
            goal_exp[i,j] = np.mean((np.outer(seqi,seqj)*contact).flatten())


    obs = pd.read_csv("./" + folder + "/" + data_out_filename + "/observables.traj", sep='\t', header=None)
    goal_sim = np.zeros((k,k))

    for i in range(k):
        for j in range(k):
            goal_sim[i,j] = np.mean(obs[i*k + j][ignore:])

    print("goals from experiment")
    print("calculated as weighted average from contact map")
    print(goal_exp)
    print("goals from simulation")
    print("calculated as mean of observables")
    print(goal_sim)


def get_diag_obs(filename):
    df = pd.read_csv(filename + "/data_out/diag_observables.traj", sep="\t", header=None)
    diag_sim = df.mean()[1:]
    diag_sim = np.array(diag_sim)
    return diag_sim

def get_diag_goal(contact, nbeads=1024, bins=16):
    diag_mask, correction = mask_diagonal(contact, bins, nbeads)
    
    plt.plot(np.log10(diag_mask*22))
    
    
    return diag_mask*22


def mask_vs_diagonal(contact, nbeads=1024, bins=16):
    diag_exp = get_diagonal(contact)
    diag_exp = downsample(diag_exp, int(nbeads/bins))
    
    diag_mask, correction = mask_diagonal(contact, bins, nbeads)
    
    plt.plot(np.log10(diag_exp) ,'--o')
    plt.plot(np.log10(diag_mask/correction), '--o')

def compare_diagonal(filename, gridsize=28.7, nbeads=1024, bins=16, plot=True, getgoal=False):

    print("diag_sim is volume fraction, average of diag_observables.traj")
    print("diag_exp is p(s), average of each subdiagonal which is then downsapled to match resolution of diagonal bins")
    print("diag_mask is mask, weighted average of contact map")
    print("corrected is mask, with correction factor to match p(s)")
    df = pd.read_csv(filename + "/diag_observables.traj", sep="\t", header=None)
    diag_sim = df.mean()[1:]
    diag_sim = np.array(diag_sim)

    if getgoal:
        df_goal = pd.read_csv(filename + "../../obj_goal_diag.txt", header=None, sep=" ")
        diag_sim_goal = df_goal.mean()[1:]
        diag_sim_goal = np.array(diag_sim_goal)

    diag_std = df.std()[1:]
    diag_std = np.array(diag_std)
    #plt.plot(diag_sim, 'o')

    contact = get_contactmap(filename + "/contacts.txt")
    diag_exp = get_diagonal(contact)
    diag_exp = downsample(diag_exp, int(nbeads/bins))

    diag_mask, correction = mask_diagonal(contact, bins, nbeads)

    #print(np.mean(diag_sim/diag_mask))

    #plt.errorbar(np.asarray(range(len(diag_sim))), np.log10(diag_sim), diag_std, '--o', label="sim")
    #plt.plot(np.log10(diag_sim/correction/22), '--o', label="volume_fraction")
    if plot:
        plt.figure(figsize=(12,10))
        plt.plot(np.log10(diag_sim), '--o', label="volume_fraction")
        plt.plot(np.log10(diag_exp), '--o', label="p(s)")
        plt.plot(np.log10(diag_mask*22), '--o', label="mask")
        #plt.plot(np.log10(diag_mask), '--o', label="mask")
        plt.plot(np.log10(diag_mask/correction), '--o', label="corrected")
        
        plt.plot(np.log10(diag_sim/22/correction), '--o', label="volfrac converted")
        if getgoal:
            plt.plot(np.log10(diag_sim_goal), label="vol frac goal")
        plt.legend()
        plt.xlabel('s')
        plt.ylabel('probability')

    #print(diag_sim_goal)
    #print(diag_sim/diag_mask)

    return diag_sim, diag_exp, diag_mask, correction



def mask_diagonal(contact, bins=16, nbeads=1024):
    rows, cols = contact.shape
    binsize = rows/bins

    measure = []
    correction = []

    for b in range(bins):
        mask = np.zeros_like(contact)
        for r in range(rows):
            for c in range(cols):
                if int((r-c)/binsize) == b:
                    mask[r,c] = 1
                    mask[c,r] = 1
        measure.append(np.mean((mask*contact).flatten()))
        correction.append(np.sum(mask)/nbeads**2)

    measure = np.array(measure)
    correction = np.array(correction)
    return measure, correction

def downsample(sequence, res):
    '''take sequence of numbers and reduce length
    res: new step size'''
    #assert(len(sequence)%res == 0)
    new = []
    for i in range(0, len(sequence), res):
        new.append(np.mean(sequence[i:i+res]))

    return np.array(new)

    

    