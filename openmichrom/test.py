import os.path as osp

from OpenMiChroM.ChromDynamics import MiChroM
from OpenMiChroM.CndbTools import cndbTools


def main():
    odir='/home/erschultz/scratch'

    idir = '/home/erschultz/dataset_11_20_23/samples/sample8'
    bead_file = osp.join(idir, 'beads.txt')


    sim = MiChroM(temperature=1.0, time_step=0.01)
    sim.setup(platform="cuda")
    sim.saveFolder(odir)
    chr10 = sim.createSpringSpiral(ChromSeq=bead_file, isRing=False)

    sim.loadStructure(chr10, center=True)

    sim.addFENEBonds(kfb=30.0)
    sim.addAngles(ka=2.0)
    sim.addRepulsiveSoftCore(Ecut=4.0)

    sim.addTypetoType(mu=3.22, rc=1.78)
    sim.addIdealChromosome(mu=3.22, rc=1.78, dinit=3, dend=500)

    sim.addFlatBottomHarmonic(kr=5*10**-3, n_rad=15.0)

    block = 3*10**2
    n_blocks = 2*10**3

    for _ in range(n_blocks):
        sim.runSimBlock(block, increment=False)

if __name__ == '__main__':
    main()
