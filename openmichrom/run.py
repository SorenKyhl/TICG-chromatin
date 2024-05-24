import csv
import os
import os.path as osp

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylib.utils import epilib
from pylib.utils.plotting_utils import plot_matrix
from scipy.optimize import curve_fit
from sklearn.preprocessing import normalize

from OpenMiChroM.ChromDynamics import MiChroM
from OpenMiChroM.CndbTools import cndbTools
from OpenMiChroM.Optimization import CustomMiChroMTraining

wd = '/home/erschultz/OpenMiChroM/Tutorials/MiChroM_Optimization'


def get_k_means_seq(y, k):
    m, _ = y.shape
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(y)
    labels = [str(i) for i in kmeans.labels_]
    labels = list(map(lambda x: x.replace('1', 'A2'), labels))
    labels = list(map(lambda x: x.replace('0', 'A1'), labels))
    return labels

def run(y, dir):
    sim = MiChroM(temperature=1.0, time_step=0.01)
    sim.setup(platform="cuda")
    sim.saveFolder(osp.join(dir, 'michrom'))

def setup():
    dir = '/home/erschultz/dataset_11_20_23/samples/sample8'
    odir = osp.join(dir, 'michrom2')
    if not osp.exists(odir):
        os.mkdir(odir, mode=0o755)
    hic_file = np.load(osp.join(dir, 'y.npy'))
    hic_file /= np.mean(hic_file.diagonal())
    np.fill_diagonal(hic_file, 1)
    m=len(hic_file)
    dense_file = osp.join(odir, 'y_dense.txt')
    np.savetxt(dense_file, hic_file)

    eigen = epilib.get_sequences(hic_file, 1, randomized=True)[0]
    labels = []
    for i, val in enumerate(eigen):
        if val > 0:
            labels.append('B1')
        else:
            labels.append('A1')

    bead_file = osp.join(odir, 'beads.txt')
    with open(bead_file, 'w', newline = '') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerows(zip(np.arange(1, m+1), labels))

    return dir, odir, bead_file, dense_file

def collpase(odir, bead_file):
    simulation = MiChroM(name='opt_chr10_100K', temperature=1.0, time_step=0.01)
    simulation.setup(platform="cuda")
    simulation.saveFolder(osp.join(odir, 'iteration_0'))
    mychro = simulation.createSpringSpiral(ChromSeq=bead_file)
    simulation.loadStructure(mychro, center=True)

    # homo
    simulation.addFENEBonds(kfb=30.0)
    simulation.addAngles(ka=2.0)
    simulation.addRepulsiveSoftCore(Ecut=4.0)

    simulation.addCustomTypes(mu=3.22, rc = 1.78, TypesTable=osp.join(wd, 'input/lambda_0'))
    simulation.addFlatBottomHarmonic(kr=5*10**-3, n_rad=8.0)

    block    = 1000
    n_blocks = 100
    rg = []
    for _ in range(n_blocks):
        simulation.runSimBlock(block, increment=False)
    rg.append(simulation.chromRG())

    #save a collapsed structure in pdb format for inspection
    simulation.saveStructure(mode = 'pdb')

    plt.plot(rg)
    plt.savefig(osp.join(odir, 'iteration_0/rg.png'))
    plt.close()

    # Remove Flat initialized in Collapse
    simulation.removeFlatBottomHarmonic()

    # Add a confinement potential with density=0.1 (volume fraction)
    simulation.addSphericalConfinementLJ()

    return simulation

def optimize(simulation, odir, bead_file, dense_file):
    input_lambdas = osp.join(wd, 'input/lambda_0')

    block    = 100
    n_blocks = 10

    for it in range(2):
        optimization = CustomMiChroMTraining(ChromSeq=bead_file,
                                         TypesTable=input_lambdas,
                                         mu=3.22, rc = 1.78)
        iteration = f"iteration_{it}"
        replica="1"
        simulation.saveFolder(osp.join(odir, iteration))
        for _ in range(n_blocks):
            # perform 1 block of simulation
            simulation.runSimBlock(block, increment=True)

            # feed the optimization with the last chromosome configuration
            optimization.prob_calculation_types(simulation.getPositions())


        with h5py.File(simulation.folder + "/Nframes_" + replica + ".h5", 'w') as hf:
            hf.create_dataset("Nframes",  data=optimization.Nframes)

        with h5py.File(simulation.folder + "/Pold_" + replica + ".h5", 'w') as hf:
            hf.create_dataset("Pold",  data=optimization.Pold)

        # Specific for Types minimization
        with h5py.File(simulation.folder + "/Pold_type_" + replica + ".h5", 'w') as hf:
            hf.create_dataset("Pold_type",  data=optimization.Pold_type)

        with h5py.File(simulation.folder + "/PiPj_type_" + replica + ".h5", 'w') as hf:
            hf.create_dataset("PiPj_type",  data=optimization.PiPj_type)

        inversion = CustomMiChroMTraining(ChromSeq=bead_file,
                                  TypesTable=input_lambdas,
                                  mu=3.22, rc = 1.78)


        with h5py.File(osp.join(odir, iteration, "Nframes_" + replica + ".h5"), 'r') as hf:
            inversion.Nframes += hf['Nframes'][()]

        with h5py.File(osp.join(odir, iteration, "Pold_" + replica + ".h5"), 'r') as hf:
            inversion.Pold += hf['Pold'][:]

        # For Types
        with h5py.File(osp.join(odir, iteration, "Pold_type_" + replica + ".h5"), 'r') as hf:
            inversion.Pold_type += hf['Pold_type'][:]

        with h5py.File(osp.join(odir, iteration, "PiPj_type_" + replica + ".h5"), 'r') as hf:
            inversion.PiPj_type += hf['PiPj_type'][:]

        lambdas = inversion.get_lambdas_types(exp_map=dense_file,
                                      damp=3e-7)

        # Probabilities of As/Bs in the simulation and experiment
        phi_sim = inversion.calc_phi_sim_types().ravel()
        phi_exp = inversion.calc_phi_exp_types().ravel()

        np.savetxt(osp.join(odir, iteration, 'phi_sim_' + iteration), phi_sim)
        np.savetxt(osp.join(odir, iteration, 'phi_exp'), phi_exp)

        plt.plot(phi_sim[phi_sim>0], 'o', label="simulation")
        plt.plot(phi_exp[phi_exp>0], 'o', label="experiment")
        plt.ylabel(r'Contact probability, $\phi$')
        labels_exp = ["AA", "AB", "BB"]
        plt.xticks(np.arange(len(labels_exp)), labels_exp)
        plt.legend()
        plt.savefig(osp.join(odir, iteration, 'phis.png'))
        plt.close()

        # Save and plot the simulated Hi-C
        dense_sim = inversion.get_HiC_sim()
        np.savetxt(osp.join(odir, iteration, 'hic_sim_' + iteration + '.dense'), dense_sim)

        dense_exp = np.loadtxt(dense_file)
        dense_exp[np.isnan(dense_exp)] = 0.0
        dense_sim /= np.mean(dense_sim.diagonal())
        r = np.zeros(dense_sim.size).reshape(dense_sim.shape)
        r = np.triu(dense_exp, k=1) + np.tril(dense_sim, k=-1) + np.diag(np.ones(len(r)))

        plot_matrix(r, osp.join(odir, iteration, 'triu.png'), vmax='mean')

        # Save the new lambda file
        lambdas_file = osp.join(odir, iteration, "lambdas")
        lambdas.to_csv(lambdas_file, index=False)
        input_lambdas = lambdas_file
        print(lambdas)

def collapse2(odir, bead_file):
    simulation_ic = MiChroM(name='opt_ic_chr10_100K', temperature=1.0, time_step=0.01)

    simulation_ic.setup(platform="cuda")
    simulation_ic.saveFolder(osp.join(odir, 'iteration_ic_0'))
    mychro_ic = simulation_ic.createSpringSpiral(ChromSeq=bead_file)
    simulation_ic.loadStructure(mychro_ic, center=True)

    # Adding Potentials section

    # **Homopolymer Potentials**
    simulation_ic.addFENEBonds(kfb=30.0)
    simulation_ic.addAngles(ka=2.0)
    simulation_ic.addRepulsiveSoftCore(Ecut=4.0)

    # **Chromosome Potentials**
    simulation_ic.addTypetoType()
    simulation_ic.addCustomIC(IClist=osp.join(wd,"input/lambda_IC_0"),
                              dinit=3, dend=200)

    # The restriction term for colapsing the beads
    simulation_ic.addFlatBottomHarmonic(kr=5*10**-3, n_rad=8.0)

    block    = 1000
    n_blocks = 200

    rg_ic = []

    for _ in range(n_blocks):
        simulation_ic.runSimBlock(block, increment=False)
        rg_ic.append(simulation_ic.chromRG())

    #save a collapsed structure in pdb format for inspection
    simulation_ic.saveStructure(mode = 'pdb')

    plt.plot(rg_ic)
    plt.savefig(osp.join(odir, 'iteration_ic_0/rg.png'))
    plt.close()

    # Remove Flat initialized in Collapse
    simulation_ic.removeFlatBottomHarmonic()

    # Add a confinement potential with density=0.1 (volume fraction)
    simulation_ic.addSphericalConfinementLJ()

    return simulation_ic

def optimize2(simulation_ic, odir, bead_file, dense_file):
    block    = 100
    n_blocks = 10
    input_ic_lambdas = osp.join(wd, "input/lambda_IC_0")
    for it in range(2):
        optimization_ic = CustomMiChroMTraining(ChromSeq=bead_file,
                                            TypesTable=osp.join(wd, 'input/lambda_0'),
                                            IClist=input_ic_lambdas,
                                            dinit=3, dend=200)
        iteration = f'iteration_ic_{it}'
        simulation_ic.saveFolder(osp.join(odir, iteration))

        for _ in range(n_blocks):
            # perform 1 block of simulation
            simulation_ic.runSimBlock(block, increment=True)

            # feed the optimization with the last chromosome configuration
            optimization_ic.prob_calculation_IC(simulation_ic.getPositions())

        replica="1"
        with h5py.File(simulation_ic.folder + "/Nframes_" + replica + ".h5", 'w') as hf:
            hf.create_dataset("Nframes",  data=optimization_ic.Nframes)

        with h5py.File(simulation_ic.folder + "/Pold_" + replica + ".h5", 'w') as hf:
            hf.create_dataset("Pold",  data=optimization_ic.Pold)

        # Specific for IC minimization
        with h5py.File(simulation_ic.folder + "/PiPj_IC_" + replica + ".h5", 'w') as hf:
            hf.create_dataset("PiPj_IC",  data=optimization_ic.PiPj_IC)

        inversion_ic = CustomMiChroMTraining(ChromSeq=bead_file,
                                     TypesTable=osp.join(wd, 'input/lambda_0'),
                                     IClist=input_lambdas,
                                     dinit=3, dend=200)

        with h5py.File(iterations + "/Nframes_" + replica + ".h5", 'r') as hf:
            inversion_ic.Nframes += hf['Nframes'][()]

        with h5py.File(iterations + "/Pold_" + replica + ".h5", 'r') as hf:
            inversion_ic.Pold += hf['Pold'][:]

        # For IC
        with h5py.File(iterations + "/PiPj_IC_" + replica + ".h5", 'r') as hf:
            inversion_ic.PiPj_IC += hf['PiPj_IC'][:]

        lambdas_ic = inversion_ic.get_lambdas_IC(exp_map="input/chr10_100k.dense",
                                         damp=5e-4)



def main():
    dir, odir, bead_file, dense_file = setup()
    simulation = collpase(odir, bead_file)
    optimize(simulation, odir, bead_file, dense_file)
    simulation_ic = collapse2(odir, bead_file)
    optimize2(simulation_ic, odir, bead_file, dense_file)

if __name__ == '__main__':
    main()
