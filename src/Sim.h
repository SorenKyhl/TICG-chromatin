#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <memory>

#include "Eigen/Dense"
#include "nlohmann/json.hpp"

#include "Bead.h"
#include "DSS_Bond.h"
#include "Harmonic_Bond.h"
#include "Harmonic_Angle.h"
#include "Cell.h"
#include "Grid.h"
#include "prof_timer.cpp"
#include "random_mars.h"

class Sim {
public:
	std::vector<Bead> beads;
	std::vector<std::unique_ptr<Bond>> bonds; // pointers because Bond class is virtual
	std::vector<std::unique_ptr<Angle>> angles; // pointers because Angle class is virtual
	Grid grid;
	std::unique_ptr<RanMars> rng;  // random number generator

	double chi;
	Eigen::MatrixXd chis;
	std::vector<double> diag_chis;
	double boundary_chi;
	double constant_chi;
	int nspecies; // number of different epigenetic marks
	int nbeads;
	double total_volume;
	double grid_size; // nm
	double bond_length; // nm
	std::string bond_type;
	double k_angle;
	float dense_diagonal_cutoff;
	float dense_diagonal_loading;
  std::string boundary_type;

	// output files
	bool redirect_stdout = false;
	FILE *xyz_out;
	FILE *energy_out;
	FILE *obs_out;
	FILE *diag_obs_out;
	FILE *constant_obs_out;
	FILE *density_out;
	FILE *extra_out;
	std::string data_out_filename;
	std::string log_filename;
	std::string xyz_out_filename;
	std::string energy_out_filename;
	std::string obs_out_filename;
	std::string diag_obs_out_filename;
	std::string constant_obs_out_filename;
	std::string density_out_filename;
	std::string contact_map_filename;
	std::string extra_out_filename;

	// MC variables
	int decay_length;
	int exp_decay;// = nbeads/decay_length;             // size of exponential falloff for MCmove second bead choice
	int exp_decay_crank;// = nbeads/decay_length;
	int exp_decay_pivot;// = nbeads/decay_length;
	double step_disp_percentage = 0.30; // step disp is this percent of bond length
	double step_trans_percentage = 0.30; // step trans is this percent of bond length
	double step_disp; // nm
	double step_trans; // nm
	double step_crank = M_PI/6; // radians 
	double step_pivot = M_PI/6; // radians
	double step_rot = M_PI/12; // radians
	double step_grid; // based off fraction of delta, see initialize

	int n_disp;
	int n_trans;
	int n_crank;
	int n_pivot;
	int n_rot;

	int acc_disp = 0;
	int acc_trans = 0;
	int acc_crank = 0;
	int acc_pivot = 0;
	int acc_rot = 0;

	bool production; // if false, equilibration
	int nSteps;// = n_trans + n_crank + n_pivot + n_rot
	int nSweeps;
	int dump_frequency; // dump every x sweeps
	int dump_stats_frequency;
	//int nEquilibSweeps; // = 10*dump_frequency;
	int acc = 0;

	bool bonded_on;
	bool nonbonded_on;
	bool angles_on;

	bool displacement_on;
	bool translation_on;
	bool crankshaft_on;
	bool pivot_on;
	bool rotate_on;

	bool gridmove_on;
	bool diagonal_on;
	bool boundary_attract_on;
	bool constant_chi_on;
	bool plaid_on;
	bool cellcount_on = true;

	bool track_contactmap;
	bool contact_bead_skipping;

	bool load_bead_types;
	bool load_configuration;
	std::string load_configuration_filename;
	std::vector<std::string> bead_type_files;

	// analytics
	bool prof_timer_on; // = false;
	bool print_trans; // = false;
	bool print_acceptance_rates; // = true;

	unsigned long nbeads_moved = 0;

	bool set_num_threads;
	int num_threads;

	std::vector<std::vector<int>> contact_map;
	int contact_resolution; //= 500;
	bool dump_density;
	bool visit_tracking;
	bool update_contacts_distance;

	bool smatrix_on;
	bool ematrix_on;
	bool dmatrix_on;

	Eigen::MatrixXd smatrix;
	std::string smatrix_filename;
	Eigen::MatrixXd ematrix;
	std::string ematrix_filename;
	Eigen::MatrixXd dmatrix;
	std::string dmatrix_filename;

	// methods
	Sim();
	Sim(std::string);
	void run();
	void xyzToContact();
	void setupContacts();
	void updateContacts();
	void updateContactsGrid();
	void updateContactsDistance();
	Eigen::MatrixXd unit_vec(Eigen::MatrixXd b);
	void readInput();
	void makeDataAndLogFiles();
	void makeOutputFiles();
	bool outside_boundary(Eigen::RowVector3d r);
	bool allBeadsInBoundary();
	void setInitialConfiguration();
	void initializeObjects();
	void calculateParameters();
	void volParameters();
	void volParameters_new();
	void loadConfiguration();
	void generateRandomCoil(double bondlength);
	int countLines(std::string filepath);
	void loadBeadTypes();
	void constructBonds();
	void constructAngles();
	void print();
	double getAllBondedEnergy();
	double getBondedEnergy(int first, int last);
	double getNonBondedEnergy(const std::unordered_set<Cell*>& flagged_cells);
	double getJustPlaidEnergy(const std::unordered_set<Cell*>& flagged_cells);
	double getJustDiagEnergy(const std::unordered_set<Cell*>& flagged_cells);
	double getJustBoundaryEnergy(const std::unordered_set<Cell*>& flagged_cells);
	double getTotalEnergy(int first, int last, const std::unordered_set<Cell*>& flagged_cells);
	double randomExp(double mu, double decay);
	void MC();
	void MCmove_displace();
	void MCmove_translate();
	void MCmove_crankshaft();
	void MCmove_rotate();
	void MCmove_pivot(int sweep);
	void MCmove_grid();
	void dumpXyz() ;
	void dumpEnergy(int sweep);
	void dumpObservables(int sweep);
	void dumpContacts(int sweep);

	void setupSmatrix();
	void setupEmatrix();
	void setupDmatrix();
};
