#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <string>
#include <sstream>
#include <cstdlib>

#include "Eigen/Dense"
#include "random_mars.cpp"
#include "nlohmann/json.hpp"
#include "prof_timer.cpp"

#include "Bead.h"
#include "Bond.h"
#include "DSS_Bond.h"

// assign variable to json key of the same name
#define READ_JSON(json, var) read_json((json), (var), #var)
template<class T>
void read_json(const nlohmann::json& json, T& var, std::string varname)
{
    std::cout << "loading " << varname << " ... ";
	if( !json.contains(varname) ) {
		throw std::runtime_error("config.contains(\""+varname+"\") has failed");
	}
	var = json[varname];
    std::cout << "loaded: " << std::to_string(var) << std::endl;
}

class Cell {
public:
	Eigen::RowVector3d r; // corner of cell... position RELATIVE TO ORIGIN... the grid origin diffuses
	std::unordered_set<Bead*> contains; // beads associated inside this gridpoint
	double vol;		                // volume of cell
	static const int beadvol = 520; // volume of a bead in the cell. 

	// number of bead types
	static int ntypes;
	std::vector<int> typenums = std::vector<int>(ntypes); // always up-to-date
	std::vector<double> phis = std::vector<double>(ntypes); // only up-to-date after energy calculation
	
	static double diag_binsize;
	static int diag_nbins;
	std::vector<double> diag_phis = std::vector<double>(diag_nbins);

	static bool diagonal_linear;

	void print() 
	{
		std::cout << r << "     N: " << contains.size() << std::endl;
		for (Bead* bead : contains)
		{
			bead->print();
		};
	}

	void reset()
	{
		// clears population trackers
		contains.clear();
		std::fill(typenums.begin(), typenums.end(), 0);  // DO NOT USE .clear()
		std::fill(phis.begin(), phis.end(), 0);      // ... it doesn't re-assign to 0's
	}

	void moveIn(Bead* bead)
	{
		// updates local number of each type of bead, but does not recalculate phis
		// TODO update populations of distance ids
		contains.insert(bead);
		for(int i=0; i<ntypes; i++)
		{
			typenums[i] += bead->d[i];
		}
	}
		
	void moveOut(Bead* bead)
	{
		// updates local number of each type of bead, but does not recalculate phis
		// TODO update populations of distance ids
		contains.erase(bead);
		for(int i=0; i<ntypes; i++)
		{
			typenums[i] -= bead->d[i];
		}
	}

	double getDensityCapEnergy()
	{
		// Density in each cell is capped at 50%
		// otherwise, incur a large energy penalty
		
		float phi_beads = contains.size()*beadvol/vol;
		float phi_solvent = 1 - contains.size()*beadvol/vol;

		double U = 0;
		if (phi_solvent < 0.5)
		{
			// high volume fraction occurs when more than 50% of the volume is occupied by beads
			U = 99999999999*phi_beads;
		}
		return U;
	}

	double getEnergy(const Eigen::MatrixXd &chis)
	{
		for (int i=0; i<ntypes; i++)
		{
			phis[i] = typenums[i]*beadvol/vol;
			//phi_solvent -= phis[i]; // wrong!! when nucl. have multiple marks
		}

		double U = 0;
		for (int i=0; i<ntypes; i++)
		{
			for (int j=i; j<ntypes; j++)
			{
				if (i==j)
				{
					// A - Solvent
					//U += chis(i,j)*phis[i]*phi_solvent*vol/beadvol;

					// A - self
					//U += -1*chis(i,j)*phis[i]*phis[j]*vol/beadvol;
					
					// the old way
					U += chis(i,j)*phis[i]*phis[j]*vol/beadvol;
					//U += exp(compressibility*( phi_beads - 0.5));
				}
				else
				{
					U += chis(i,j)*phis[i]*phis[j]*vol/beadvol;
					//U += exp(compressibility*( phi_beads - 0.5));
					//std::cout << chis(i,j) << " " << phis[i] << " " << phis[j] << " " << vol << " " << beadvol << std::endl;
				}
			}
		}
		return U;
	}

	double getDiagEnergy(const std::vector<double> diag_chis)
	{
		for (int i=0; i<diag_nbins; i++)
		{
			diag_phis[i] = 0;
		}

		double Udiag = 0;
		//int index;
		//for (Bead* bead1 : contains)
		//{
			//for (Bead* bead2 : contains)
			//{
				//index = std::floor( abs(bead1->id - bead2->id) / diag_binsize);
				//assert (index >= 0);
				//assert (index <= diag_nbins);
				//diag_phis[index] += 1; // diag phis is just a count, multiply by volumes later
			//}
		//}
		int d_index; // genomic separation (index for diag_phis)
		int imax = (int) contains.size();
		std::vector<int> indices;
		for (const auto& elem : contains)
		{
			indices.push_back(elem->id);
		}

		// count pairwise contacts 
		for (int i=0; i<imax-1; i++)
		{
			for(int j=i+1; j<imax; j++)
			{
				d_index  = std::floor( abs(indices[i] - indices[j]) / diag_binsize);
				diag_phis[d_index] += 1; // diag phis is just a count, multiply by volumes later
			}
		}

		for (int i=0; i<diag_nbins; i++)
		{
			diag_phis[i] *= beadvol/vol; // convert to actual volume fraction

			if (diagonal_linear) {
				Udiag += diag_chis[i]*diag_phis[i];
			}
			else {
				Udiag += diag_chis[i]* diag_phis[i]*diag_phis[i];
			}
		}

		// multiply by vol/beadvol to calculate mean-field energy
		// needs to be different for linear case?
		//if(!diagonal_linear) { Udiag *= vol/beadvol;}
		return Udiag*vol/beadvol; 
	}

	double getBoundaryEnergy(const double boundary_chi, const double delta)
	{
		// TODO: this is broken if the grid is moving; 
		// need to check relative to origin
		double Uboundary = 0;
		for (const auto& bead : contains)
		{
			if (bead->r(0) < delta)
			{
				Uboundary += boundary_chi;
			}
		}
		return Uboundary;
	}
};

class Grid {
public:
	std::vector<std::vector<std::vector<Cell>>> cells; 
	std::unordered_set<Cell*> active_cells;       // cells marked as active (within simulation region)

	double delta;              // grid cell size (length)

	bool cubic_boundary = true;
	bool spherical_boundary = false;

	int L;                        // size of cubic boundary in units of grid cells
	double radius;                // radius of simulation volume in [nanometers]
	int boundary_radius;          // radius of boundary in units of grid cells
	Eigen::RowVector3d sphere_center; // center of spherical boundary

	// origin is the bottom-left-most grid cell for cubic simulations
	// With grid moves on, it will diffuse with periodic boundaries
	// inside the volume bounded by (-delta, -delta, -delta) and (0,0,0) 
	Eigen::RowVector3d origin; 

	void generate()
	{
		origin = {-delta/2.0,-delta/2.0,-delta/2.0};

		// need L+1 grid cells per direction to fully enclose simulation boundary
		std::cout << "Initializing cells: " << std::endl;
		int cells_per_dim = L+1;
		cells.resize(cells_per_dim);
		for(int i=0; i<cells_per_dim; i++) {
			cells[i].resize(cells_per_dim);
			for(int j=0; j<cells_per_dim; j++) {
				cells[i][j].resize(cells_per_dim);
				for(int k=0; k<cells_per_dim; k++) {
					
					cells[i][j][k].vol = delta*delta*delta;
					cells[i][j][k].r = {i*delta,j*delta,k*delta}; // position relative to origin
					//cells[i][j][k].print();
				}
			}
		}
	}

	void setActiveCells()
	{
		std::cout << "Setting active cells" << std::endl;
		for(int i=0; i<=L; i++) {
			for(int j=0; j<=L; j++) {
				for(int k=0; k<=L; k++) {

					if (cubic_boundary)
					{
						// all cells are active at all times
						active_cells.insert(&cells[i][j][k]);
					}
					else if (spherical_boundary)
					{
						// only cells within sqrt(3)*delta of radius are active, at all times
						Eigen::RowVector3d cell_corner;
						cell_corner(0) = origin(0) + i*delta;
						cell_corner(1) = origin(1) + j*delta;
						cell_corner(2) = origin(2) + k*delta;

						Eigen::RowVector3d difference = cell_corner - sphere_center;

						if (difference.norm() < radius + sqrt(3)*delta)
						{
							active_cells.insert(&cells[i][j][k]);
						}
					}
				}
			}
		}
	}
 	
	void printActiveCells()
	{
		// prints only the active cells that contains particles
		std::cout << "Printing Active Cells: number:" << active_cells.size() << std::endl;
		for(Cell* cell : active_cells) {
			if(cell->contains.size() > 0) cell->print();
		}
	}

	void meshBeads(std::vector<Bead> &beads)
	{
		// Inserts all beads into their corresponding grid cells
		for(int i=0; i<cells.size(); i++) {
			for(int j=0; j<cells[i].size(); j++) {
				for(int k=0; k<cells[i][j].size(); k++) {
					cells[i][j][k].reset(); // initialize cells to contain no beads
				}
			}
		}

		int i, j, k;
		for(Bead& bead : beads)
		{
			i = floor((bead.r(0) - origin(0))/delta);
			j = floor((bead.r(1) - origin(1))/delta);
			k = floor((bead.r(2) - origin(2))/delta);

			cells[i][j][k].moveIn(&bead);
		}
	}

	Cell* getCell(const Bead& bead)
	{
		// Returns a pointer to the cell in which a bead is located
		int i, j, k;
		i = floor((bead.r(0) - origin(0))/delta);
		j = floor((bead.r(1) - origin(1))/delta);
		k = floor((bead.r(2) - origin(2))/delta);
		return &cells[i][j][k];
	}

	Cell* getCell(const Eigen::RowVector3d &r)
	{
		// Returns a pointer to the cell in which a bead is located
		int i, j, k;
		i = floor((r(0) - origin(0))/delta);
		j = floor((r(1) - origin(1))/delta);
		k = floor((r(2) - origin(2))/delta);
		return &cells[i][j][k];
	}

	bool checkCellConsistency(int nbeads)
	{
		// checks to see if number of beads in the grid is the same as simulation nbeads
		double cellbeads = 0;
		for(Cell* cell : active_cells)
		{
			cellbeads += cell->contains.size();
		}

		return (cellbeads == nbeads); 
	}

	double densityCapEnergy(const std::unordered_set<Cell*>& flagged_cells)
	{
		// energy penalty due to density cap
		double U = 0; 
		for(Cell* cell : flagged_cells)
		{
			U += cell->getDensityCapEnergy();
		}
		return U;
	}
			
	double energy(const std::unordered_set<Cell*>& flagged_cells, const Eigen::MatrixXd &chis)
	{
		// nonbonded volume interactions
		double U = 0; 
		for(Cell* cell : flagged_cells)
		{
			U += cell->getEnergy(chis);
		}
		return U;
	}

	double diagEnergy(const std::unordered_set<Cell*>& flagged_cells, const std::vector<double> diag_chis)
	{
		// nonbonded volume interactions
		double U = 0; 
		for(Cell* cell : flagged_cells)
		{
			U += cell->getDiagEnergy(diag_chis);
		}
		return U;
	}

	double boundaryEnergy(const std::unordered_set<Cell*>& flagged_cells, const double boundary_chi)
	{
		// nonbonded volume interactions
		double U = 0; 
		for(Cell* cell : flagged_cells)
		{
			U += cell->getBoundaryEnergy(boundary_chi, delta);
		}
		return U;
	}

	double get_ij_Contacts(int i, int j) 
	{
		// calculates average phi_i phi_j
		double obs  = 0;
		for(Cell* cell : active_cells)
		{
			obs += cell->phis[i] * cell->phis[j];
		}

		obs /= active_cells.size(); 
		return obs;
	}

	void getDiagObs(std::vector<double> &diag_obs)
	{
		for (Cell* cell : active_cells)
		{
			for(int i=0; i<diag_obs.size(); i++)
			{
				if (Cell::diagonal_linear) {
					diag_obs[i] += cell->diag_phis[i];
				}
				else {
					diag_obs[i] += cell->diag_phis[i] * cell->diag_phis[i];
				}
			}
		}

		for (int i=0; i<diag_obs.size(); i++)
		{
			diag_obs[i] /= active_cells.size();
		}
	}

	double cellCount()
	{
		int sum = 0;
		int num = 0;
		for (Cell* cell : active_cells)
		{
			sum += cell->contains.size();
			num++;
		}

		return (double) sum / (double) num;
	}
};


class Sim {
public: 
	std::vector<Bead> beads;
	std::vector<Bond*> bonds; // pointers because Bond class is virtual
	Grid grid;
	RanMars* rng;  // random number generator

	double chi; 
	Eigen::MatrixXd chis;
	std::vector<double> diag_chis;
	double boundary_chi;
	int nspecies; // number of different epigenetic marks
	int nbeads; 
	double total_volume;
	double grid_size;

	// output files
	FILE *xyz_out; 
	FILE *energy_out;
	FILE *obs_out;
	FILE *diag_obs_out;
	FILE *density_out;

	// MC variables
	int decay_length; 
	int exp_decay;// = nbeads/decay_length;             // size of exponential falloff for MCmove second bead choice
	int exp_decay_crank;// = nbeads/decay_length;
	int exp_decay_pivot;// = nbeads/decay_length;
	double step_disp = 5;
	double step_trans = 2;
	double step_crank = M_PI/6;
	double step_pivot = M_PI/6;
	double step_rot = M_PI/12;
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

	bool displacement_on;
	bool translation_on;
	bool crankshaft_on;
	bool pivot_on;
	bool rotate_on;

	bool gridmove_on;
	bool diagonal_on;
	bool boundary_attract_on;
	bool plaid_on;
	bool cellcount_on = true;


	bool track_contactmap;
	bool load_chipseq;
	bool load_configuration;
	std::string load_configuration_filename; 
	std::vector<std::string> chipseq_files;
	
	// analytics
	bool print_MC; // = false;
	bool print_trans; // = false;
	bool print_acceptance_rates; // = true;

	unsigned long nbeads_moved = 0;

	std::vector<std::vector<int>> contact_map;
	int contact_resolution; //= 500;
	bool dump_density;
	bool visit_tracking;
	bool update_contacts_distance;

	void setupContacts()
	{
		std::cout << "setting up contacts" << std::endl;
		int nbins = nbeads/contact_resolution + 1;
		contact_map.resize(nbins);
		for(int i=0; i<nbins; i++)
		{
			contact_map[i].resize(nbins);
			for(int j=0; j<nbins; j++) 
			{
				contact_map[i][j] = 0;
			}
		}
	}

	void updateContacts()
	{
		if (update_contacts_distance){
			updateContactsDistance();
		}
		else {
			updateContactsGrid();
		}
	}

	void updateContactsGrid()
	{
		// contact defined as residing within same grid cell
		{
			int rows = contact_map.size();
			int cols = contact_map.size();
			int id1, id2;
			std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));


			for(Cell* cell : grid.active_cells)
			{
				for(Bead* bead1 : cell->contains)
				{
					for(Bead* bead2 : cell->contains)
					{
						id1 = bead1->id/contact_resolution;
						id2 = bead2->id/contact_resolution;

						if (visited[id1][id2] == false)
						{
							contact_map[id1][id2] += 1;

							if (visit_tracking) visited[id1][id2] = true;
						}
					}
				}
			}
		}
	}

	void updateContactsDistance()
	{
		// contacts defined as two beads within a cutoff distance
		double cutoff = 28.7; // nm
		int id1, id2;
		for(int i=0; i<beads.size(); i++)
		{
			for(int j=i; j<beads.size(); j++)
			{
				Eigen::RowVector3d distance =  beads[i].r - beads[j].r;
				if (distance.norm() < cutoff)
				{
					id1 = beads[i].id/contact_resolution;
					id2 = beads[j].id/contact_resolution;
					contact_map[id1][id2] += 1;
					if (id1 != id2) contact_map[id2][id1] += 1;
				}
			}
		}
	}


	Eigen::MatrixXd unit_vec(Eigen::MatrixXd b)
	{
		// generates random unit vector
		double R1,R2,R3;
		do { R1 = (2*rng->uniform()-1);
			R2 = (2*rng->uniform()-1);
			R3 = R1*R1+R2*R2;
		} while (R3>=1);

		b(0) = 2*sqrtl(1.0-R3)*R1;
		b(1) = 2*sqrtl(1.0-R3)*R2;
		b(2) = 1-2.0*R3;
		return b;
	}

	void readInput()
	{
		std::cout << "reading input file ... " << std::endl;

		std::ifstream i("config.json");
		nlohmann::json config;
		i >> config;

		assert(config.contains("plaid_on")); plaid_on = config["plaid_on"];
		
		if (plaid_on)
		{
			assert(config.contains("nspecies")); nspecies = config["nspecies"];
			assert(config.contains("load_chipseq")); load_chipseq = config["load_chipseq"];

			if (load_chipseq)
			{
				chis = Eigen::MatrixXd::Zero(nspecies, nspecies); 

				char first = 'A' + 1;
				for (int i=0; i<nspecies; i++)
				{

					// should be included even if load_chipseq is false... fix later

					for (int j=i; j<nspecies; j++)
					{
						char first = 'A' + i;
						char second = 'A' + j;
						std::string chistring = "chi";
						chistring += first;
						chistring += second;
						chis(i,j) = config[chistring];         //  i must be less than j
						std::cout << chistring << " " << chis(i,j) << std::endl;
					}
				}

				assert(config.contains("chipseq_files"));
				for (auto file : config["chipseq_files"])
				{
					chipseq_files.push_back(file);
				}


				if (chipseq_files.size() != nspecies)
				{
					throw std::runtime_error("Number of chipseq files: " + std::to_string(chipseq_files.size()) + " must equal number of species: " + std::to_string(nspecies));
				}
				Cell::ntypes = nspecies;
			}
		}
		else
		{
			nspecies = 0;
			load_chipseq = false;
		}

		assert(config.contains("diagonal_on")); diagonal_on= config["diagonal_on"];
		assert(config.contains("nbeads")); nbeads = config["nbeads"];

		if (diagonal_on)
		{
			assert(config.contains("diag_chis"));
			for (auto e : config["diag_chis"])
			{
				diag_chis.push_back(e);
			}

			Cell::diag_nbins = diag_chis.size();
			std::cout << "number of bins: " << Cell::diag_nbins << std::endl;
			Cell::diag_binsize = nbeads / diag_chis.size();
			std::cout << "binsize " << Cell::diag_binsize << std::endl;
		}

		assert(config.contains("gridmove_on")); gridmove_on = config["gridmove_on"];
		//
		std::cout << "grid move is : " << gridmove_on << std::endl;
		//assert(config.contains("production")); production = config["production"];
		READ_JSON(config, production); 
		assert(config.contains("decay_length")); decay_length = config["decay_length"];
		assert(config.contains("nSweeps")); nSweeps = config["nSweeps"];
		assert(config.contains("dump_frequency")); dump_frequency = config["dump_frequency"];
		assert(config.contains("dump_stats_frequency")); dump_stats_frequency = config["dump_stats_frequency"];
		assert(config.contains("bonded_on")); bonded_on = config["bonded_on"];
		assert(config.contains("nonbonded_on")); nonbonded_on = config["nonbonded_on"];
		assert(config.contains("displacement_on")); displacement_on = config["displacement_on"];
		assert(config.contains("translation_on")); translation_on = config["translation_on"];
		assert(config.contains("crankshaft_on")); crankshaft_on = config["crankshaft_on"];
		assert(config.contains("pivot_on")); pivot_on = config["pivot_on"];
		assert(config.contains("rotate_on")); rotate_on = config["rotate_on"];
		assert(config.contains("load_configuration")); load_configuration = config["load_configuration"];
		assert(config.contains("load_configuration_filename")); load_configuration_filename = config["load_configuration_filename"];
		assert(config.contains("print_MC")); print_MC = config["print_MC"];
		assert(config.contains("print_trans")); print_trans = config["print_trans"];
		assert(config.contains("print_acceptance_rates")); print_acceptance_rates = config["print_acceptance_rates"];
		assert(config.contains("contact_resolution")); contact_resolution = config["contact_resolution"];
		assert(config.contains("grid_size")); grid_size = config["grid_size"];
		assert(config.contains("track_contactmap")); track_contactmap = config["track_contactmap"];
		assert(config.contains("diagonal_linear")); Cell::diagonal_linear = config["diagonal_linear"];
		assert(config.contains("dump_density")); dump_density = config["dump_density"];
		assert(config.contains("visit_tracking")); visit_tracking = config["visit_tracking"];
		assert(config.contains("update_contacts_distance")); update_contacts_distance = config["update_contacts_distance"];
		assert(config.contains("boundary_attract_on")); boundary_attract_on = config["boundary_attract_on"];
		assert(config.contains("boundary_chi")); boundary_chi  = config["boundary_chi"];

		//cellcount_on = config["cellcount_on"];

		assert(config.contains("seed"));
		int seed = config["seed"];
		rng = new RanMars(seed);

		std::cout << "read successfully" << std::endl;
	}

	void makeOutputFiles()
	{
		std::cout << "making output files ... ";

		// make and populate data output directory
		const int dir_err = system("mkdir data_out");
		if (dir_err == -1) {std::cout << "error making data_out directory" << std::endl;}
	        xyz_out = fopen("./data_out/output.xyz", "w");
		energy_out = fopen("./data_out/energy.traj", "w");
		obs_out = fopen("./data_out/observables.traj", "w");
		diag_obs_out = fopen("./data_out/diag_observables.traj", "w");
		density_out = fopen("./data_out/density.traj", "w");

		std::cout << "created successfully" << std::endl;
	}

	bool outside_boundary(Eigen::RowVector3d r)
	{
		bool is_out = false;

		if (grid.cubic_boundary)
		{
			is_out = (r.minCoeff() < 0 || r.maxCoeff() > grid.L*grid.delta);
		}
		else if (grid.spherical_boundary)
		{
			is_out = r.norm() > grid.boundary_radius*grid.delta;
		}

		return is_out;
	}

		
	void initialize()
	{
		std::cout << "Initializing simulation objects ... " << std::endl;
		Timer t_init("Initializing");

		grid.delta = grid_size;
		std::cout << "grid size is : " << grid.delta << std::endl;
		step_grid = grid.delta/10.0; // size of grid displacement MC moves

		double Vbar = 7765.77;  // nm^3/bead: reduced number volume per spakowitz: V/N
		double vol = Vbar*nbeads; // simulation volume in nm^3
		grid.L= std::round(std::pow(vol,1.0/3.0) / grid.delta); // number of grid cells per side // ROUNDED, won't exactly equal a desired volume frac
		std::cout << "grid.L is: " << grid.L << std::endl;
		total_volume = pow(grid.L*grid.delta/1000.0, 3); // micrometers^3 ONLY TRUE FOR CUBIC SIMULATIONS 
		std::cout << "volume is: " << total_volume << std::endl;

		grid.radius = std::pow(3*vol/(4*M_PI), 1.0/3.0); // radius of simulation volume
		grid.boundary_radius = std::round(grid.radius); // radius in units of grid cells
		// sphere center needs to be centered on a multiple of grid delta
		//grid.sphere_center = {grid.boundary_radius*grid.delta, grid.boundary_radius*grid.delta, grid.boundary_radius*grid.delta};
		grid.origin = {grid.boundary_radius*grid.delta, grid.boundary_radius*grid.delta, grid.boundary_radius*grid.delta};

		exp_decay = nbeads/decay_length;             // size of exponential falloff for MCmove second bead choice
		exp_decay_crank = nbeads/decay_length;
		exp_decay_pivot = nbeads/decay_length;

		n_disp = displacement_on ? nbeads : 0;
		n_trans = translation_on ? decay_length : 0; 
		n_crank = crankshaft_on ? decay_length : 0;
		n_pivot = pivot_on ? decay_length/10: 0;
		n_rot = rotate_on ? nbeads : 0;
		nSteps = n_trans + n_crank + n_pivot + n_rot;

		double bondlength = 16.5;
		beads.resize(nbeads);  // uses default constructor initialization to create nbeads;

		std::cout << "load configuratin is " << load_configuration << std::endl;
		// set configuration
		if(load_configuration) {

			std::cout << "Loading configuration from " << load_configuration_filename << std::endl;
			std::ifstream IFILE; 
			IFILE.open(load_configuration_filename); 
			std::string line;
			getline(IFILE, line); // nbeads line
			std::cout << line << std::endl;
			getline(IFILE, line); // comment line 
			std::cout << line << std::endl;
				
			// first bead
			getline(IFILE, line);
			//std::cout << line << std::endl;
			std::stringstream ss;
			ss << line;
			ss >> beads[0].id;
			ss >> beads[0].r(0);
			ss >> beads[0].r(1);
			ss >> beads[0].r(2);

			for(int i=1; i<nbeads; i++)
			{
				getline(IFILE, line);
				//std::cout << line << std::endl;
				std::stringstream ss;  // new stream so no overflow from last line
				ss << line;

				ss >> beads[i].id;
				ss >> beads[i].r(0);
				ss >> beads[i].r(1);
				ss >> beads[i].r(2);

				beads[i-1].u = beads[i].r - beads[i-1].r;
				beads[i-1].u = beads[i-1].u.normalized();
			}
			beads[nbeads-1].u = unit_vec(beads[nbeads-1].u); // random orientation for last bead
			
		}
		else {
			// RANDOM COIL 
			double center; // center of simulation box
			if (grid.cubic_boundary)
			{
				center = grid.delta*grid.L/2;
			}
			else if (grid.spherical_boundary)
			{
				center = 0;
			}

			beads[0].r = {center, center, center}; // start in middlle of the box
			beads[0].u = unit_vec(beads[0].u);
			beads[0].id = 0;

			for(int i=1; i<nbeads; i++)
			{
				do {
					beads[i].u = unit_vec(beads[i].u); 
					beads[i].r = beads[i-1].r + bondlength*beads[i].u; // orientations DO NOT point along contour
					beads[i].id = i;
				} while (outside_boundary(beads[i].r));
			}
		}

		// set up chipseq
        if (load_chipseq) {
            int marktype = 0;
            for (std::string chipseq_file : chipseq_files)
            {
                std::ifstream IFCHIPSEQ;
                int tail_marked;
                IFCHIPSEQ.open(chipseq_file);
                for(int i=0; i<nbeads; i++)
                {
					//beads[i].d.reserve(nspecies);
					beads[i].d.resize(nspecies);
                    IFCHIPSEQ >> tail_marked;
                    if (tail_marked == 1)
                    {
                        beads[i].d[marktype] = 1;
                    }
                    else
                    {
                        beads[i].d[marktype] = 0;
                    }
                }
                marktype++;
                IFCHIPSEQ.close();
            }
        }

		// set bonds
		bonds.resize(nbeads-1); // use default constructor
		for(int i=0; i<nbeads-1; i++)
		{
			bonds[i] = new DSS_Bond{&beads[i], &beads[i+1]};
		}

		dumpData();
		std::cout << "Objects created" << std::endl;
	}

	void print()
	{
		std::cout << "simulation in : "; 
		std::cout << "With beads: " << std::endl;
		for(Bead& bb : beads) bb.print();             // use reference to avoid copies
		std::cout << "And bonds: " << std::endl;
		for(Bond* bo : bonds) bo->print();
	}

	double getAllBondedEnergy()
	{
		double U = 0;
		for(Bond* bond : bonds) {U += bond->energy();}
		return U;
	}

	double getBondedEnergy(int first, int last)
	{
		double U = 0;
		//for(Bond* bo : bonds) U += bo->energy();  // inefficient 
		if (first>0) U += bonds[first-1]->energy(); // move affects bond going into first
		if (last<(nbeads-1)) U += bonds[last]->energy();   // ... and leaving the second
		return U;
	}

	double getNonBondedEnergy(const std::unordered_set<Cell*>& flagged_cells)
	{
		// gets all the nonbonded energy
		auto start = std::chrono::high_resolution_clock::now();

		double U = grid.densityCapEnergy(flagged_cells);
		if (plaid_on)
		{
			U += grid.energy(flagged_cells, chis);
		}
		if (diagonal_on)
		{
			U += grid.diagEnergy(flagged_cells, diag_chis); 
		}
		if (boundary_attract_on)
		{
			U += grid.boundaryEnergy(flagged_cells, boundary_chi);
		}

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
		//std::cout << "NonBonded took " << duration.count() << "microseseconds "<< std::endl;
		return U;
	}

	double getJustDiagEnergy(const std::unordered_set<Cell*>& flagged_cells)
	{
		// for when dumping energy; 
		double U = grid.diagEnergy(flagged_cells, diag_chis); 
		return U;
	}

	double getJustBoundaryEnergy(const std::unordered_set<Cell*>& flagged_cells)
	{
		// for when dumping energy; 
		double U = grid.boundaryEnergy(flagged_cells, boundary_chi); 
		return U;
	}

	double getTotalEnergy(int first, int last, const std::unordered_set<Cell*>& flagged_cells)
	{
		double U = 0;
		if (bonded_on) U += getBondedEnergy(first, last);
		if (nonbonded_on) U += getNonBondedEnergy(flagged_cells);
		return U;
	}

	double randomExp(double mu, double decay)
	{
		// generates random number distributed according to two-sided exponential distribution
		// centered about mu, with characteristic decay length
		double cdf_y;
		do {
			cdf_y = rng->uniform();
		} while (cdf_y <= 0); // cdf_y cannot exactly equal 0, otherwise inverse cdf is -infty 


		if (cdf_y > 0.5) {
			return mu - decay*log(1 - 2*abs(cdf_y - 0.5)); // inverse cdf
		}
		else {
			return mu + decay*log(1 - 2*abs(cdf_y - 0.5)); // inverse cdf
		}
	}

	void MC()
	{
		std::cout << "Beginning Simulation" << std::endl;
		for(int sweep = 0; sweep<nSweeps; sweep++)
		{
			//std::cout << sweep << std::endl; 
			double nonbonded;
			//nonbonded = getNonBondedEnergy(grid.active_cells);
			//std::cout << "beginning sim: nonbonded: " <<  grid.active_cells.size() << std::endl;

		    looping:
			Timer t_translation("translating", print_MC);
			for(int j=0; j<n_trans; j++)
			{
				MCmove_translate();
				//nonbonded = getnonbondedenergy(grid.active_cells);
				//std::cout << nonbonded << std::endl;
			}
			t_translation.~Timer();

			if (gridmove_on) MCmove_grid();
			//nonbonded = getNonBondedEnergy(grid.active_cells);
			//std::cout << nonbonded << std::endl;

			Timer t_displace("displacing", print_MC);
			for(int j=0; j<n_disp; j++)
			{
				MCmove_displace();
				//nonbonded = getNonBondedEnergy(grid.active_cells);
				//std::cout << nonbonded << std::endl;
			}
			t_displace.~Timer();


			Timer t_crankshaft("Cranking", print_MC);
			for(int j=0; j<n_crank; j++) {
				MCmove_crankshaft();
				//nonbonded = getNonBondedEnergy(grid.active_cells);
				//std::cout << nonbonded << std::endl;
			}
			t_crankshaft.~Timer();

			
			Timer t_rotation("Rotating", print_MC);
			for(int j=0; j<n_rot; j++) {
				MCmove_rotate();
			}
			t_rotation.~Timer();
			

			Timer t_pivot("pivoting", print_MC);
			for(int j=0; j<n_pivot; j++) {
				MCmove_pivot(sweep);
				//nonbonded = getNonBondedEnergy(grid.active_cells);
				//std::cout << nonbonded << std::endl;
			}
			t_pivot.~Timer();


			if (sweep%dump_frequency == 0) {
				std::cout << "Sweep number " << sweep << std::endl;
				dumpData();
				
				if (print_acceptance_rates) {
					std::cout << "acceptance rate: " << (float) acc/((sweep+1)*nSteps)*100.0 << "%" << std::endl;

					if (displacement_on) std::cout << "disp: " << (float) acc_disp/((sweep+1)*n_disp)*100 << "% \t";
					if (translation_on) std::cout << "trans: " << (float) acc_trans/((sweep+1)*n_trans)*100 << "% \t";
					if (crankshaft_on) std::cout << "crank: " << (float) acc_crank/((sweep+1)*n_crank)*100 << "% \t";
					if (pivot_on) std::cout << "pivot: " << (float) acc_pivot/((sweep+1)*n_pivot)*100 << "% \t";
					if (rotate_on) std::cout << "rot: " << (float) acc_rot/((sweep+1)*n_rot)*100 << "% \t"; 
					//std::cout << "cellcount: " << grid.cellCount();
					std::cout << std::endl;
					
				}

				if (production) {dumpContacts(sweep);}
			}

			if (sweep%dump_stats_frequency == 0)
			{
				if (production)
				{
					updateContacts();  // calculate contact data, but dont dump to file
					dumpObservables(sweep);
				}

				Timer t_allenergy("all energy", print_MC);

				double bonded = getAllBondedEnergy();
				double nonbonded = 0;
				nonbonded = nonbonded_on ? getNonBondedEnergy(grid.active_cells) : 0; // includes diagonal and boundary energy
				double diagonal = 0;
				diagonal = diagonal_on ? getJustDiagEnergy(grid.active_cells) : 0;
				double boundary = 9;
				boundary = boundary_attract_on ? getJustBoundaryEnergy(grid.active_cells) : 0;
				//std::cout << "bonded " << bonded << " nonbonded " << nonbonded << std::endl;
				t_allenergy.~Timer();

				dumpEnergy(sweep, bonded, nonbonded, diagonal, boundary);
			}

		}

		// final contact map
		dumpContacts(nSweeps);
		std::cout << "acceptance rate: " << (float) acc/(nSweeps*nSteps)*100.0 << "%" << std::endl;
	}

	
	void MCmove_displace()
	{
	    Timer t_displacemove("Displacement move", print_MC);
		// pick random particle
		int o = floor(beads.size()*rng->uniform());

		// copy old info (don't forget orientation, etc)
		Cell* old_cell = grid.getCell(beads[o]);
		
		Eigen::RowVector3d displacement;
		displacement = step_disp*unit_vec(displacement);

		Eigen::RowVector3d new_location = beads[o].r + displacement;

		// check if exited the simulation box, if so reject the move
		if (outside_boundary(new_location))
		{
			return;
		}

		Cell* new_cell = grid.getCell(new_location);

		std::unordered_set<Cell*> flagged_cells;
		flagged_cells.insert(old_cell);
		flagged_cells.insert(new_cell);

		double Uold = getTotalEnergy(o, o, flagged_cells);
		
		// move
		beads[o].r = new_location;

		// check if moved grid into new grid cell, update grid
		if (new_cell != old_cell)
		{
			new_cell->moveIn(&beads[o]);
			old_cell->moveOut(&beads[o]);
		}

		double Unew = getTotalEnergy(o, o, flagged_cells);

		if (rng->uniform() < exp(Uold-Unew))
		{
			//std::cout << "Accepted"<< std::endl;
			acc += 1;
			acc_disp += 1;
			nbeads_moved += 1;
		}
		else
		{
			//std::cout << "Rejected" << std::endl;
			beads[o].r -= displacement ;
			new_cell->moveOut(&beads[o]);
			old_cell->moveIn(&beads[o]);
		}
	}

	void MCmove_translate()
	{
		if (print_trans) std::cout << "==================NEW MOVE ====================" << std::endl;
		Timer t_trans("Translate", print_trans);
		//Timer t_setup("setup", print_trans);

		// select a first bead at random
		int first = floor(beads.size()*rng->uniform());

		// choose second bead from two-sided exponential distribution around first
		int last = -1; 
		while (last < 0 || last >= nbeads)
		{
			last = round(randomExp(first, exp_decay));            // does this obey detailed balance?
		}

		// swap first and last to ensure last > first
		if (last < first) {std::swap(first, last);} 
		
		if (print_trans) std::cout << "number of beads is " << last - first << std::endl;

		// generate displacement vector with magnitude step_trans
		Eigen::RowVector3d displacement;
		displacement = step_trans*unit_vec(displacement);

		// memory storage objects
		std::unordered_set<Cell*> flagged_cells;
		std::unordered_map<int, std::pair<Cell*, Cell*>> bead_swaps; // index of beads that swapped cell locations
	
		flagged_cells.reserve(last-first);
		bead_swaps.reserve(last-first);

		Cell* old_cell_tmp;
		Cell* new_cell_tmp;
		Eigen::RowVector3d new_loc;

		//t_setup.~Timer();

		// execute move
		try
		{
			//Timer t_bounds("bounds", print_trans);
			// reject immediately if moved out of simulation box, no cleanup necessary
			for(int i=first; i<=last; i++)
			{
				new_loc = beads[i].r + displacement;
				if (outside_boundary(new_loc)) {
					throw "exited simulation box";	
				}
			}
			//t_bounds.~Timer();

			//Timer t_flag("Flag cells", print_trans);
			// flag appropriate cells for energy calculation and find beads that swapped cells
			for(int i=first; i<=last; i++)
			{
				old_cell_tmp = grid.getCell(beads[i]);
				flagged_cells.insert(old_cell_tmp);

				new_loc = beads[i].r + displacement;
				new_cell_tmp = grid.getCell(new_loc);

				if (new_cell_tmp != old_cell_tmp)
				{
					bead_swaps[i] = std::make_pair(old_cell_tmp, new_cell_tmp);
					flagged_cells.insert(new_cell_tmp);
				}
			}
			//t_flag.~Timer();

			//Timer t_uold("Uold", print_trans);
			//std::cout << "Beads: " << last-first << " Cells: " << flagged_cells.size() << std::endl;
			double Uold = getTotalEnergy(first, last, flagged_cells);
			//t_uold.~Timer();

			//Timer t_disp("Displacement", print_trans);
			for(int i=first; i<=last; i++)
			{
				beads[i].r += displacement;
			}
			//t_disp.~Timer();

			//Timer t_swap("Bead Swaps", print_trans);
			// update grid <bead index,   <old cell , new cell>>
			//for(std::pair<int, std::pair<Cell*, Cell*>> &x : bead_swaps)
			for(auto const &x : bead_swaps)
			{
				x.second.first->moveOut(&beads[x.first]);
				x.second.second->moveIn(&beads[x.first]);
			}
			//t_swap.~Timer();

			//Timer t_unew("Unew", print_trans);
			double Unew = getTotalEnergy(first, last, flagged_cells);
			//t_unew.~Timer();

			if (rng->uniform() < exp(Uold-Unew))
			{
				acc += 1;
				acc_trans += 1;
				nbeads_moved += (last-first);
			}
			else
			{
				throw "rejected";
			}
		}
		// REJECTION CASES -- restore old conditions
		catch (const char* msg)
		{
			Timer t_rej("rejection", print_trans); 
			if(msg == "rejected")
			{
				// restore particle positions 
				for(int i=first; i<=last; i++)
				{
					beads[i].r -= displacement;
				}
				
				if (bead_swaps.size() > 0)
				{
					//for(std::pair<int, std::pair<Cell*, Cell*>> &x : bead_swaps)
					for(auto const &x : bead_swaps)
					{
						x.second.first->moveIn(&beads[x.first]);
						x.second.second->moveOut(&beads[x.first]);
					}
				}
			}
		}
	}

	void MCmove_crankshaft()
	{
		// index of first bead through index of last bead represent all beads that MOVE in the crankshaft
		// axis of crankshaft motion is between beads (first-1) and (last+1), because those two do not move.

		// select a first bead at random (cannot be first or last bead)
		int first = -1;
		while (first < 1 || first > nbeads-2)
		{
			first = floor(beads.size()*rng->uniform());
		}

		// choose second bead from two-sided exponential distribution around first
		int last = -1; 
		while (last < 1 || last > nbeads-2)
		{
			last = round(randomExp(first, exp_decay_crank));
		}

		// swap first and last to ensure last > first
		if (last < first) {std::swap(first, last);} 

		// compute axis of rotation, create quaternion 
		Eigen::RowVector3d axis = beads[last+1].r - beads[first-1].r;
                double angle = step_crank*(rng->uniform()- 0.5); // random symmtric angle in cone size step_crank
		Eigen::Quaterniond du;
                du = Eigen::AngleAxisd(angle, axis.normalized()); // object representing this rotation

		// memory storage objects
		std::vector<Eigen::RowVector3d> old_positions;
		std::vector<Eigen::RowVector3d> old_orientations;
		std::unordered_set<Cell*> flagged_cells;
		std::unordered_map<int, std::pair<Cell*, Cell*>> bead_swaps; 

		old_positions.reserve(last-first);
		old_orientations.reserve(last-first);
		flagged_cells.reserve(last-first);
		bead_swaps.reserve(last-first);

		Cell* old_cell_tmp;
		Cell* new_cell_tmp;

		// execute move
		try
		{
			double Uold = 0;
			if(bonded_on) Uold += getBondedEnergy(first, last);

			for(int i=first; i<=last; i++)
			{
				// save old configuration
				// --------------------- can this be done more efficiently? ------------------------------------------
				old_positions.push_back(beads[i].r);
				old_orientations.push_back(beads[i].u);

				// step to new configuration, but don't update grid yet (going to check if in bounds first)
				beads[i].r = du*(beads[i].r - beads[first-1].r) + beads[first-1].r.transpose();
				beads[i].u = du*beads[i].u; 
			}

			// reject if moved out of simulation box, need to restore old bead positions
			for(int i=first; i<=last; i++)
			{
				if (outside_boundary(beads[i].r))
				{
					throw "exited simulation box";	
				}
			}

			// flag cells and bead swaps, but do not update the grid 
			for(int i=first; i<=last; i++)
			{
				new_cell_tmp = grid.getCell(beads[i]);
				old_cell_tmp = grid.getCell(old_positions[i-first]);

				flagged_cells.insert(old_cell_tmp);

				if (new_cell_tmp != old_cell_tmp)
				{
					flagged_cells.insert(new_cell_tmp);
					bead_swaps[i] = std::make_pair(old_cell_tmp, new_cell_tmp);
				}
			}

			// calculate old nonbonded energy based on flagged cells
			if (nonbonded_on) Uold += getNonBondedEnergy(flagged_cells);

			
			// Update grid
			//for(std::pair<int, std::pair<Cell*, Cell*>> &x : bead_swaps)
			for(auto const &x : bead_swaps)
			{
				x.second.first->moveOut(&beads[x.first]); // out of the old cell
				x.second.second->moveIn(&beads[x.first]); // in to the new cell
			} 

			double Unew = getTotalEnergy(first, last, flagged_cells);

			if (rng->uniform() < exp(Uold-Unew))
			{
				//std::cout << "Accepted"<< std::endl;
				acc += 1;
				acc_crank += 1;
				nbeads_moved += (last-first);
			}
			else
			{
				//std::cout << "Rejected" << std::endl;
				throw "rejected";
			}
		}
		// REJECTION CASES -- restore old conditions
		catch (const char* msg)
		{
			// restore particle positions 
			for(int i=0; i<old_positions.size(); i++)
			{
				beads[first+i].r = old_positions[i];
				beads[first+i].u = old_orientations[i];
			}
			
			// restore grid allocations
			if (bead_swaps.size() > 0)
			{
				//for(std::pair<int, std::pair<Cell*, Cell*>> &x : bead_swaps)
				for(auto const &x : bead_swaps)
				{
					x.second.first->moveIn(&beads[x.first]);   // back in to the old
					x.second.second->moveOut(&beads[x.first]); // back out of the new
				}
			}
		}
	}

	void MCmove_rotate()
	{
		// pick random particle
		int o = floor(beads.size()*rng->uniform());

                // old configuration
		Eigen::RowVector3d u_old = beads[o].u;

		double Uold = getBondedEnergy(o, o); // only need bonded energy for roation moves

                // step to new configuration
                double angle = step_rot*(rng->uniform()- 0.5); // random symmtric angle in cone size step_rot
		Eigen::RowVector3d axis;                     // random axis
                axis = unit_vec(axis); 
		Eigen::Quaterniond du {Eigen::AngleAxisd(angle, axis)}; // object representing this rotation

		beads[o].u = du*beads[o].u;

		double Unew = getBondedEnergy(o, o);

		if (rng->uniform() < exp(Uold-Unew))
		{
			//std::cout << "Accepted"<< std::endl;
			acc += 1;
			acc_rot += 1;
		}
		else
		{
			//std::cout << "Rejected" << std::endl;
			beads[o].u = u_old;
		}
        }

	void MCmove_pivot(int sweep)
	{
		// index terminology:
		// pivot === the bead being pivoted around, but not itself moved
		// [first,last] === the interval of beads physically moved in the pivot
	        // depending on which end the pivot is executed, this is either [0, pivot-1] or [pivot+1, nbeads-1]
		// The only bonds affected are pivot-1 OR pivot;

		// chose one end of the polymer and a pivot bead
		int end = (nbeads-1)*round(rng->uniform()); //  either first bead or last bead

		end = nbeads-1;

		// pick second bead according to single-sided exponential distribution away from end
		int length;
		do {
			length = abs(round(randomExp(0, exp_decay_pivot))); // length down the polymer from the end
		} while (length < 1 || length > nbeads-1);

		int pivot = (end == 0) ? length : (nbeads-1-length);

		int first = (pivot < end) ? pivot+1 : end;
		int last = (pivot < end) ? end : pivot-1;

		// rotation objects
                double angle = step_pivot*(rng->uniform()- 0.5); // random symmtric angle in cone size step_pivot 
		Eigen::RowVector3d axis;                     // random axis
                axis = unit_vec(axis); 
		Eigen::Quaterniond du {Eigen::AngleAxisd(angle, axis)}; // object representing this rotation

		// memory storage objects
		std::vector<Eigen::RowVector3d> old_positions;
		std::vector<Eigen::RowVector3d> old_orientations;
		std::unordered_set<Cell*> flagged_cells;
		std::unordered_map<int, std::pair<Cell*, Cell*>> bead_swaps;

		Cell* old_cell_tmp;
		Cell* new_cell_tmp;

		// execute move
		try
		{
			double Uold = 0;
			if(bonded_on) Uold += getBondedEnergy(pivot-1, pivot);

			for(int i=first; i<=last; i++)
			{
				// save old positions
				old_positions.push_back(beads[i].r);
				old_orientations.push_back(beads[i].u);

				// step to new configuration, but don't update grid yet (going to check if in bounds first)
				beads[i].r = du*(beads[i].r - beads[pivot].r) + beads[pivot].r.transpose();
				beads[i].u = du*beads[i].u; 
			}

			// reject if moved out of simulation box
			for(int i=first; i<=last; i++)
			{
				if (outside_boundary(beads[i].r))
				{
					throw "exited simulation box";	
				}
			}

			// flag cells and bead swaps, but do not update the grid 
			for(int i=first; i<=last; i++)
			{
				new_cell_tmp = grid.getCell(beads[i]);
				old_cell_tmp = grid.getCell(old_positions[i-first]);

				flagged_cells.insert(old_cell_tmp);

				if (new_cell_tmp != old_cell_tmp)
				{
					flagged_cells.insert(new_cell_tmp);
					bead_swaps[i] = std::make_pair(old_cell_tmp, new_cell_tmp);
				}
			}

			// calculate old nonbonded energy based on flagged cells
			if(nonbonded_on) Uold += getNonBondedEnergy(flagged_cells);
			
			// Update grid
			//for(std::pair<int, std::pair<Cell*, Cell*>> &x : bead_swaps)
			for(auto const &x : bead_swaps)
			{
				x.second.first->moveOut(&beads[x.first]); // out of the old cell
				x.second.second->moveIn(&beads[x.first]); // in to the new cell
			} 

			double Unew = getTotalEnergy(pivot-1, pivot, flagged_cells);


			if (rng->uniform() < exp(Uold-Unew))
			{
				//std::cout << "Accepted"<< std::endl;
				acc += 1;
				acc_pivot += 1;
				nbeads_moved += (last-first);
			}
			else
			{
				//std::cout << "Rejected" << std::endl;
				throw "rejected";
			}
		}
		// REJECTION CASES -- restore old conditions
		catch (const char* msg)
		{
			// restore particle positions 
			for(int i=0; i<old_positions.size(); i++)
			{
				beads[first+i].r = old_positions[i];
				beads[first+i].u = old_orientations[i];
			}
			
			// restore bead allocations
			if (bead_swaps.size() > 0)
			{
				//for(std::pair<int, std::pair<Cell*, Cell*>> &x : bead_swaps)
				for(auto const &x : bead_swaps)
				{
					x.second.first->moveIn(&beads[x.first]);   // back in to the old
					x.second.second->moveOut(&beads[x.first]); // back out of the new
				}
			}
		}

	}

	void MCmove_grid()
	{
		// not really a MC move (metropolis criterion doesn't apply) 
		// don't need to choose active cells; they are chosen at the beginning of the
		// simulation to include all cells that could possibly include particles.
		bool flag = true;
		double U;
		while (flag)
		{
			Eigen::RowVector3d displacement;
			Eigen::RowVector3d old_origin = grid.origin;

			displacement = step_grid*unit_vec(displacement);
			grid.origin += displacement;

			// periodic boundary conditions: move inside volume bounded by (-delta, -delta, -delta) and (0,0,0)
			grid.origin(0) -= std::ceil(grid.origin(0) / grid.delta) * grid.delta;
			grid.origin(1) -= std::ceil(grid.origin(1) / grid.delta) * grid.delta;
			grid.origin(2) -= std::ceil(grid.origin(2) / grid.delta) * grid.delta;

			// remesh beads.
			grid.meshBeads(beads);

			U = getNonBondedEnergy(grid.active_cells);

			// don't accept if move violates density maximum
			if (U < 9999999999)
			{ 
				//std::cout << "passed grid move" << std::endl;
				flag = false;
			}
			else
			{
				//std::cout << "failed grid move" << std::endl;
				flag = false;
				grid.origin = old_origin;
				grid.meshBeads(beads); // remesh back with old origin
			}
		}
	}

	
	void dumpData() 
	{ 
		xyz_out = fopen("./data_out/output.xyz", "a");
		fprintf(xyz_out, "%d\n", nbeads);
		fprintf(xyz_out, "atoms\n");

		for(Bead bead : beads)
		{
			fprintf(xyz_out, "%d\t %lf\t %lf\t %lf\t" , bead.id,bead.r(0),bead.r(1),bead.r(2));

			if (plaid_on)
			{
				for(int i=0; i<nspecies; i++)
				{
					fprintf(xyz_out, "%d\t", bead.d[i]);
				}
			}

			fprintf(xyz_out, "\n");
		}
		fclose(xyz_out); 
	}

	void dumpEnergy(int sweep, double bonded=0, double nonbonded=0, double diagonal=0, double boundary=0)
	{
		energy_out = fopen("./data_out/energy.traj", "a");
		fprintf(energy_out, "%d\t %lf\t %lf\t %lf\t %lf\n", sweep, bonded, nonbonded, diagonal, boundary);
		fclose(energy_out);
	}

	void dumpObservables(int sweep)
	{
		if (plaid_on)
		{
			obs_out = fopen("./data_out/observables.traj", "a");
			fprintf(obs_out, "%d", sweep);

			for (int i=0; i<nspecies; i++)
			{
				for (int j=i; j<nspecies; j++)
				{
					double ij_contacts = grid.get_ij_Contacts(i, j);
					fprintf(obs_out, "\t%lf", ij_contacts);
				}
			}

			fprintf(obs_out, "\n");
			fclose(obs_out);
		}

		if (diagonal_on)
		{
			diag_obs_out = fopen("./data_out/diag_observables.traj", "a");
			fprintf(diag_obs_out, "%d", sweep);

			std::vector<double> diag_obs(diag_chis.size(), 0.0);
			grid.getDiagObs(diag_obs);

			for(auto& e : diag_obs)
			{
				fprintf(diag_obs_out, "\t%lf", e);
			}

			fprintf(diag_obs_out, "\n");
			fclose(diag_obs_out);
		}

		if (dump_density)
		{
			density_out = fopen("./data_out/density.traj", "a");
			fprintf(density_out, "%d", sweep);

			double avg_density = 0;
			int i = 0;
			for (Cell* cell : grid.active_cells)
			{
				i++;
				//fprintf(density_out, " %lf", cell->phis[0]);
				avg_density += cell->phis[0];
			}
			avg_density /= i;
			fprintf(density_out, " %lf\n", avg_density);
			fclose(density_out);
		}

	}

	void dumpContacts(int sweep)
	{
		std::string contact_map_filename;
		if (track_contactmap)
		{
			// outputs new contact map, doesn't override
			contact_map_filename = "./data_out/contacts" + std::to_string(sweep) + ".txt";
		}
		else
		{
			// overwrites contact file with most current values
			contact_map_filename = "./data_out/contacts.txt";
		}

		std::ofstream contactsOutFile(contact_map_filename);
		for (const auto &row : contact_map) {
			for (const int &element : row) {
				contactsOutFile << element << " ";
			}
			contactsOutFile << "\n";
		}
	}

	void run()
	{
		readInput();
		makeOutputFiles();
		initialize(); 
		grid.generate();  // creates the grid locations
		grid.meshBeads(beads);  // populates the grid locations with beads;
		grid.setActiveCells();  // populates the active cell locations
		setupContacts();
		MC(); // MC simulation
		assert (grid.checkCellConsistency(nbeads));
	}
};

//TOOD: phase out
int Cell::ntypes;
int Cell::diag_nbins;
double Cell::diag_binsize;
bool Cell::diagonal_linear;

int main()
{
	auto start = std::chrono::high_resolution_clock::now();

	Sim mySim;
	mySim.run();

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
	std::cout << "Took " << duration.count() << "seconds "<< std::endl;
	std::cout << "Moved " << mySim.nbeads_moved << " beads " << std::endl;
	return 0;
}
