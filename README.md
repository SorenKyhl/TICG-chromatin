#TICG-chromatin

all dependencies included

build process:
make && mv TICG-engine sample/

execute:
cd sample && ./TICG-engine

requires config.json to specify simulation parameters
see sample/ for reference

config parameters:
seed: seed for random number generator
production: immediately begins recording data, without equilibration period
nspecies: number of distinct bead labels
decay_length: determines how many beads are moved in translation, crankshaft, and pivot moves. A block of beads is chosen according to an exponential with characteristic size nbeads/decay_length. e.g. if nbeads=1000 and decay_length=10, a block will be chosen with characteristic size 1000/10 = 100. 
nSweeps: number of monte carlo sweeps. each sweep executes $nbeads displacements, $nbeads $decay_length translations, $decay_length crankshafts, $decay_length/10 pivots, and $nbeads rotations.
dump_frequency: [units = sweeps] every $dump_frequency sweeps, the sweep number and acceptance rates will be sent to stdout, an .xyz configuration will be appended to output.xyz, and contacts.txt will be updated. memory intensive, so dump_frequency should be less frequent than dump_stats_frequency
dump_stats_frequency: [units = sweeps] every $dump_stats_frequency sweeps, energies will be calculated and output to energy.traj and observables output to observables.traj
nEquilibSweeps: nubmer of equilibrium sweeps, if $production is false. During equilibration sweeps, energy.traj and observables.traj are not updated.
bonded_on: polymer chain bonds on
nonbonded_on: all nonbonded interactions on
binding_on: deprecated
diagonal_on: diagonal nonbonded interactions on
plaid_on: plaid nonbonded interactions on (chiAA, chiAB, etc)
displacement_on: single bead displacement MC moves on. Every sweep executes $nbeads displacements
translation_on: tranlsation MC moves on. Every sweep executes $decay_length translations, in which a random block of beads is translated together. The size of the block is determined by decay_length, according to an exponential probability function
crankshaft_on: crankshaft MC moves on. Every sweep executes $decay_length crankshafts, in which a random block of beds is rotated about the axis defined by the first and last bead. The size of the block is determined by decay_length, according to an exponential probability distribution.
pivot_on: pivot MC moves on. Every sweep executes $decay_length/10 pivot moves, in which a random end of the chain is chosen, and then a block of beads extending from that end inwards is pivoted about the axis of the interior-most bond of the block. The size of the block is determined by decay_length, according to an exponential probability distribution.
AB_block: deprecated. 
domainsize: deprecated
load_chiseq: assign bead labls according to the 1-d vectors in $chipseq_files.
load_configuration: initialize the system according to the <input>.xyz file specified by $load_configuration_filename
load_configuration_filename: initial configuration to start the simulation, if $load_configuration is on
prof_timer_on: enables timers, to profile the speed of MC moves. NOTE: Timers are commented out and will not work as of Aug 8, 2021 because they cause a double free bug when running on midway.. but not on local machine TODO: figure this out.
print_trans: enables profile timers for translate MC moves
print_acceptance_rates: if true, prints acceptance rates of each type of MC move to stout
contact resolution: [units = beads] size of contact_map pixels in units of beads.
gridmove_on: if true, will displace the origin of the TICG grid every sweep to avoid affects of a static grid.
diag_chis: [units=kT] array of energy parameters for diagonal interaction. 
grid_size: [units=nm] size of TICG grid cells. 
diagonal_linear: particular functional of diagnoal interactions. leave true
visit_tracking: deprecated
dump_density: deprecated
update_condacts_distance: calculate contact map based off pairwise distances instead of TICG grid. Define contact as within some radius rather than within the same TICG cell
boundary_attract_on: cells adjacent to the xy plane have an attractive energy for all beads set by $boundary_chi
boundary_chi: strength of boundary attraction, if $boundary_attract_on 
chi<XY>: Flory-huggins type energy of interaction between beads labeled X and Y.
chipseq_files: list of chipseq files to assign bead labels
