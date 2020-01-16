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
#include <Eigen/Dense>
#include <random_mars.cpp>
#include <nlohmann/json.hpp>
#include "prof_timer.cpp"

unsigned long nbeads_moved = 0;
//RanMars rng(1);

class Cell; 

int test;

class Tail {
public:
	Tail(bool m)
		: mark{m}, bound{0} {}
	
	static constexpr double eps_methylated = -0.01;   // kT
	static constexpr double eps_unmethylated = 1.52;  // kT

	bool mark; // methylation state: me3?
	bool bound; // bound by HP1?

	double energy(double chem)
	{
		// calculates binding energy as a function of chemical potential (chem = kT log[HP1_free])
		return bound*(mark*(eps_methylated-chem) + !mark*(eps_unmethylated-chem));
	}

	void flipBind(Cell* cell_inside);

	void print()
	{
		std::cout <<  "Tail with epigenetic state: " << mark << " binding state " << bound << std::endl;
	}
};


class Bead {
public:
	Bead(int i, double x=0, double y=0, double z=0, bool mark1=0, bool mark2=0)
		: id{i}, r{x,y,z}, tail1{mark1}, tail2{mark2} {}
	
	Bead() 
		: id{0}, r{0,0,0}, tail1{0}, tail2{0} {}


	//enum Domain {A, B, C, D, E, F, G, H};
	//int d; // domain

	int ntypes = 7;
	std::vector<int> d = std::vector<int>(ntypes);

	int id;   // i don't think i need this anymore
	Eigen::RowVector3d r;
	Eigen::RowVector3d u;

	Tail tail1;
	Tail tail2; 
	int tails_methylated;

	static constexpr double Jprime = -4; // kT

	double tailEnergy(double chem)
	{
		//    intranucleosome HP1 interaction |        HP1 binding energy
		return Jprime*tail1.bound*tail2.bound + tail1.energy(chem) + tail2.energy(chem);
	}

	int nbound()
	{
		// number of bound HP1
		return tail1.bound + tail2.bound;
	}

	void print() {std::cout << id <<" "<< r << std::endl;}
};


// abstract class
class Bond {
public:
	Bond(Bead* b1, Bead* b2)
		: pbead1{b1}, pbead2{b2} {}

	Bead* pbead1;
	Bead* pbead2;

	void print() {std::cout << pbead1->id <<" "<< pbead2->id << std::endl;}
	virtual double energy() = 0; 
};


class DSS_Bond : public Bond {
public:
	DSS_Bond(Bead* bead1, Bead* bead2) 
		: Bond{bead1, bead2} {}

        double delta;
        double ata;
        double gamma;
        double eps_bend;
        double eps_parl;
        double eps_perp;

	double energy()
	{
		// DELTA = 1??
		/*
		double delta = 1;
		double ata =  2.7887;
		double gamma = 0.83281;
		double eps_bend = 1.4668;
		double eps_parl = 34.634;
		double eps_perp = 16.438;
		*/

		// DELTA = 0.1
		/*
		double delta = 0.1;
		double ata = 21.651;
		double gamma = 0.98168;
		double eps_bend = 1.5515;
		double eps_parl = 818.08;
		double eps_perp = 1107.8;
		*/

		// DELTA = 0.33 
		double delta = 16.5;    // dimless (is it 16.5 or 0.33????)
		double ata = 0.152;        // nm-1
		double gamma = 0.938;      // dimless
		double eps_bend = 78.309;  // kT nm
		double eps_parl = 2.665;   // kT/nm
		double eps_perp = 1.942;   // kT/nm

		double U = 0;

		Eigen::RowVector3d R = pbead2->r - pbead1->r;              
		Eigen::RowVector3d Rparl = R.dot(pbead1->u)*pbead1->u;
		Eigen::RowVector3d Rperp = R - Rparl;

		// checks:
		//cout << "-------" << endl;
		//cout << "R is :         " << R << endl;
		//cout << "R|_ + R||   =  " << Rparl + Rperp << endl;
		//cout << "R|_ dot R|| =  " << Rparl.dot(Rperp) << endl;
		//cout << "R|| norm is   : " << Rparl.norm() << endl;
		//cout << "R|| norm is   : " << R.dot(pbead1->u) << endl;

		//U += eps_bend*(u.row(i) - u.row(i-1) - ata*Rperp).squaredNorm();  // bend energy
		//U += eps_parl*pow((R.dot(u.row(i-1)) - delta*gamma), 2);               // stretch energy
		//U += eps_perp*Rperp.squaredNorm();                                // shear energy

		U += eps_bend*(pbead2->u - pbead1->u - ata*Rperp).squaredNorm();
		U += eps_parl*pow((R.dot(pbead1->u) - delta*gamma), 2);
		U += eps_perp*Rperp.squaredNorm(); 
		U /= 2*delta;

		return U;
	}
};


class Harmonic_Bond : public Bond {
public:
	Harmonic_Bond(Bead* bead1, Bead* bead2, double kk, double r00) 
		: Bond{bead1, bead2}, k{kk}, r0{r00} {}

	double k;
	double r0;

	double energy()
	{
		Eigen::RowVector3d displacement = pbead2->r - pbead1->r; 
		double r = sqrt(displacement.dot(displacement));
		return k*(r- r0)*(r - r0); // can be optimized if r0 is zero
	}
};


class Cell {
public:
	Eigen::RowVector3d r; // corner of cell?????
	std::unordered_set<Bead*> contains; // beads associated inside this gridpoint
	double phi;
	double vol;
	int local_HP1; // local number of HP1
	static const int beadvol = 520; // volume of a bead in the cell. 
	static constexpr double vint = 4/3*M_PI*3*3*3; // volume of HP1 interaction
	static constexpr double J = -4;  // kT 

	const int ntypes = 7;
	std::vector<int> typenums = std::vector<int>(ntypes);
	std::vector<double> phis = std::vector<double>(ntypes);

	void print() 
	{
		std::cout << r << "     N: " << contains.size() << std::endl;
		for (Bead* bead : contains)
		{
			std::cout << "With beads: " ;
			bead->print();
		};
	}

	void moveIn(Bead* bead)
	{
		contains.insert(bead);
		local_HP1 += bead->nbound();
		//typenums += bead->d;

		for(int i=0; i<ntypes; i++)
		{
			typenums[i] += bead->d[i];
		}
	}
		
	void moveOut(Bead* bead)
	{
		contains.erase(bead);
		local_HP1 -= bead->nbound();
		//typenums -= bead->d;

		for(int i=0; i<ntypes; i++)
		{
			typenums[i] -= bead->d[i];
		}
	}

	double getEnergy(const Eigen::MatrixXd &chis)
	{
		// nonbonded volume interactions
		//phi = contains.size()*beadvol/vol;
		//if(phi > 0.5) {phi = 99999999;}
		//double U = vol*phi*phi/beadvol;

		// AB volume interactions
		//assert (contains.size() == A_contains.size() + B_contains.size());
		//double phiA = typenums[0]*beadvol/vol;
		//double phiB = typenums[1]*beadvol/vol;

		//bool high_volfrac = false;

		//int total_beads = 0;
		//for (int n : typenums)
		//{
			//total_beads += n; wrong, this is the total number of marks, not beads.
		//}
		//float phi_solvent = 1 - total_beads*vol/beadvol;
		float phi_solvent = 1 - contains.size()*beadvol/vol;

		for (int i=0; i<ntypes; i++)
		{
			phis[i] = typenums[i]*beadvol/vol;
			//phi_solvent -= phis[i]; // wrong!! when nucl. have multiple marks
		}

		//if (phiA > 0.5 || phiB > 0.5) {phiA = 999999999;}

		double U = 0;

		if (phi_solvent < 0.1 )
		{
			// high volume fraction occurs when more than 50% of the volume is occupied by beads
			U = 99999999999;
		}
		else 
		{
			for (int i=0; i<ntypes; i++)
			{
				for (int j=i; j<ntypes; j++)
				{
					if (i==j)
					{
						U += chis(i,j)*phis[i]*phi_solvent*vol/beadvol;
						//std::cout << chis(i,j) << " " << phis[i] << " " << phi_solvent << " " << vol << " " << beadvol << std::endl;
					}
					else
					{
						U += chis(i,j)*phis[i]*phis[j]*vol/beadvol;
						//std::cout << chis(i,j) << " " << phis[i] << " " << phis[j] << " " << vol << " " << beadvol << std::endl;
					}
				}
			}
			//std::cout << "inside else statement" << U << std::endl;
		}

		//double Uold = chis[0]*vol*phiA*phiB/beadvol + chis[1]*vol*phiA*phiA/beadvol + chis[2]*vol*phiB*phiB/beadvol;
		return U;
	}

	double getTailEnergy()
	{
		double rho = local_HP1/vol;
		double U = J/2*vint*vol*rho*rho;
		return  U;
	}
};


class Grid {
public:
	std::vector<std::vector<std::vector<Cell>>> cells; 
	std::unordered_set<Cell*> active_cells;       // cells marked as active (within simulation region)

	double delta;              // grid cell size 
	int L;                     // size of cubic boundary in units of grid cells
	int boundary_radius;          // radius of boundary in units of grid cells

	// bottom-left-most grid cell: 
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
				for(int k=0; k<L; k++) {
					
					cells[i][j][k].phi = 0.0;
					cells[i][j][k].vol = delta*delta*delta;
					cells[i][j][k].r = {i*delta,j*delta,k*delta};
					//cells[i][j][k].print();
				}
			}
		}
	}

	void setActiveCells()
	{
		// cubic simulation boundary
		std::cout << "Setting active cells" << std::endl;
		for(int i=0; i<=L; i++) {
			for(int j=0; j<=L; j++) {
				for(int k=0; k<=L; k++) {
					active_cells.insert(&cells[i][j][k]);
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
		int i, j, k;
		for(Bead& bead : beads)   // the reference is crucial-- otherwise, copies are made in the for-each loop
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

	bool checkHp1Consistency(int hp1_tot);
			
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

	double tailEnergy(const std::unordered_set<Cell*>& flagged_cells)
	{
		// HP1 oligomerization volume interactions
		double U = 0;
		for(Cell* cell : flagged_cells)
		{
			U += cell->getTailEnergy();
		}
		return U;
	}

	unsigned long get_AB_Contacts() {
		unsigned long n = 0;
		for(Cell* cell : active_cells)
		{
			//n += cell->typenums[Bead::A] * cell->typenums[Bead::B];
			n += cell->typenums[0] * cell->typenums[1];
		}
		return n;
	}

	unsigned long get_AA_Contacts() {
		unsigned long n = 0;
		for(Cell* cell : active_cells)
		{
			//n += cell->typenums[Bead::A] * cell->typenums[Bead::A];
			n += cell->typenums[0] * cell->typenums[0];
		}
		return n;
	}

	unsigned long get_BB_Contacts() {
		unsigned long n = 0;
		for(Cell* cell : active_cells)
		{
			//n += cell->typenums[Bead::B] * cell->typenums[Bead::B];
			n += cell->typenums[1] * cell->typenums[1];
		}
		return n;
	}

	unsigned long get_ij_Contacts(int i, int j) 
	{
		unsigned long n = 0;
		for(Cell* cell : active_cells)
		{
			n += cell->typenums[i] * cell->typenums[j];
		}
		return n;
	}
};


class Sim {
public: 
	std::vector<Bead> beads;
	std::vector<Bond*> bonds; // pointers because Bond class is virtual
	Grid grid;

	RanMars* rng; 

	double chi; // = 1;  
	Eigen::MatrixXd chis;
	int nspecies; // number of different epigenetic marks
	int nbeads; // = 32000;// 362918; 
	double hp1_mean_conc; // = 285;  // [uM]
	double hp1_total; // conserved quantity 
	double total_volume;
	static int hp1_free;

	bool cubic_boundary = true;
	bool spherical_boundary = false;

	FILE *xyz_out; 
	FILE *energy_out;
	FILE *obs_out;

	// MC variables
	int decay_length; // = 1000;
	int exp_decay;// = nbeads/decay_length;             // size of exponential falloff for MCmove second bead choice
	int exp_decay_crank;// = nbeads/decay_length;
	int exp_decay_pivot;// = nbeads/decay_length;
	double step_disp = 1;
	double step_trans = 2;
	double step_crank = M_PI/6;
	double step_pivot = M_PI/6;
	double step_rot = M_PI/12;

	int n_disp;// = 0;
	int n_trans;// = decay_length; 
	int n_crank;// = decay_length;
	int n_pivot;// = decay_length;
	int n_bind;// = 0;
	int n_rot;// = nbeads;

	int acc_trans = 0;
	int acc_crank = 0;
	int acc_pivot = 0;
	int acc_rot = 0;
	int acc_bind = 0;

	bool production; // if false, equilibration
	int nSteps;// = n_trans + n_crank + n_pivot + n_rot + n_bind;
	int nSweeps; //100000;
	int dump_frequency; // = 100; // dump every x sweeps
	int dump_stats_frequency;
	//int nEquilibSweeps; // = 10*dump_frequency;
	int acc = 0;

	bool bonded_on; // = true;
	bool nonbonded_on; //= true;
	bool binding_on; // = false; 

	bool AB_block; // = true;
	int domainsize = nbeads;

	bool load_chipseq; // = false;
	bool load_configuration; // = true;
	std::string load_configuration_filename; // = "32k_input.xyz";
	std::string load_chipseq_filename; // = "chrom16_H3K9me3.txt";
	std::vector<std::string> chipseq_files;

	std::string chipseq_1;
	std::string chipseq_2;
	std::string chipseq_3;
	std::string chipseq_4;
	std::string chipseq_5;
	std::string chipseq_6;
	std::string chipseq_7;
	
	// analytics
	bool print_MC; // = false;
	bool print_trans; // = false;
	bool print_acceptance_rates; // = true;

	std::vector<std::vector<int>> contact_map;
	int contact_resolution; //= 500;

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
		for(Cell* cell : grid.active_cells)
		{
			for(Bead* bead1 : cell->contains)
			{
				for(Bead* bead2 : cell->contains)
				{
					int id1 = bead1->id/contact_resolution;
					int id2 = bead2->id/contact_resolution;
					contact_map[id1][id2] += 1;
				}
			}
		}
	}

	double getChemPot()
	{
		// chemical potential based on [HP1_free] in micromolar. 
		// Conversion ratio: (HP1_free/um^3) / [uM] = 602
		return log(hp1_free/total_volume / (602));
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
		std::cout << "reading input file ... ";

		std::ifstream i("config.json");
		nlohmann::json config;
		i >> config;
		
		nspecies = config["nspecies"];
		load_chipseq = config["load_chipseq"];

		if (load_chipseq)
		{
			chis = Eigen::MatrixXd::Zero(nspecies, nspecies); 

			char first = 'A' + 1;
			for (int i=0; i<nspecies; i++)
			{

				// should be included even if load_chipseq is false... fix later

				for (int j=i; j<nspecies; j++)
				{
					//std::cout << i << ", " << j << std::endl;
					char first = 'A' + i;
					char second = 'A' + j;
					//std::cout << first << ", " << second << std::endl;
					std::string chistring = "chi";
					chistring += first;
					chistring += second;
					std::cout << chistring << std::endl;
					chis(i,j) = config[chistring];
					std::cout << chistring << " " << chis(i,j) << std::endl;
				}
			}

			chipseq_1 = config["chipseq_1"];
			chipseq_files.push_back(chipseq_1);
			chipseq_2 = config["chipseq_2"];
			chipseq_files.push_back(chipseq_2);
			chipseq_3 = config["chipseq_3"];
			chipseq_files.push_back(chipseq_3);
			chipseq_4 = config["chipseq_4"];
			chipseq_files.push_back(chipseq_4);
			chipseq_5 = config["chipseq_5"];
			chipseq_files.push_back(chipseq_5);
			chipseq_6 = config["chipseq_6"];
			chipseq_files.push_back(chipseq_6);
			chipseq_7 = config["chipseq_7"];
			chipseq_files.push_back(chipseq_7);
		}
	
		std::cout << "made it out" << std::endl;
		production = config["production"];
		nbeads = config["nbeads"];
		//chi = config["chi"];
		//chis[0] = config["chiAB"];
		//chis[1] = config["chiA"];
		//chis[2] = config["chiB"];
		hp1_mean_conc = config["hp1_mean_conc"];
		decay_length = config["decay_length"];
		nSweeps = config["nSweeps"];
		dump_frequency = config["dump_frequency"];
		dump_stats_frequency = config["dump_stats_frequency"];
		bonded_on = config["bonded_on"];
		nonbonded_on = config["nonbonded_on"];
		binding_on = config["binding_on"];
		AB_block = config["AB_block"];
		domainsize = config["domainsize"];
		load_configuration = config["load_configuration"];
		load_configuration_filename = config["load_configuration_filename"];
		load_chipseq_filename = config["load_chipseq_filename"];
		print_MC = config["print_MC"];
		print_trans = config["print_trans"];
		print_acceptance_rates = config["print_acceptance_rates"];
		contact_resolution = config["contact_resolution"];

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

		std::cout << "created successfully" << std::endl;
	}

	bool outside_boundary(Eigen::RowVector3d r)
	{
		bool is_out = false;

		if (cubic_boundary)
		{
			is_out = (r.minCoeff() < 0 || r.maxCoeff() > grid.L*grid.delta);
		}
		else if (spherical_boundary)
		{
			is_out = r.norm() > grid.boundary_radius*grid.delta;
		}

		return is_out;
	}

		
	void initialize()
	{
		std::cout << "Initializing simulation objects ... " << std::endl;
		Timer t_init("Initializing");

		grid.delta=28.7; // grid cell size nm

		double Vbar = 7765.77;  // nm^3/bead: reduced number volume per spakowitz: V/N
		grid.L= std::round(std::pow(nbeads*Vbar,1.0/3.0) / grid.delta); // number of grid cells per side // ROUNDED, won't exactly equal a desired volume frac
		std::cout << "grid.L is: " << grid.L << std::endl;
		total_volume = pow(grid.L*grid.delta/1000.0, 3); // micrometers^3 

		grid.boundary_radius = 7;

		hp1_free = hp1_mean_conc*total_volume* 602;
		hp1_total = hp1_free;

		exp_decay = nbeads/decay_length;             // size of exponential falloff for MCmove second bead choice
		exp_decay_crank = nbeads/decay_length;
		exp_decay_pivot = nbeads/decay_length;

		n_disp = 0;
		n_trans = decay_length; 
		n_crank = decay_length;
		n_pivot = decay_length;
		n_bind = nbeads;
		n_rot = nbeads;
		nSteps = n_trans + n_crank + n_pivot + n_rot + n_bind;

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
			std::cout << line << std::endl;
			std::stringstream ss;
			ss << line;
			ss >> beads[0].id;
			ss >> beads[0].r(0);
			ss >> beads[0].r(1);
			ss >> beads[0].r(2);

			for(int i=1; i<nbeads; i++)
			{
				getline(IFILE, line);
				std::cout << line << std::endl;
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
			if (cubic_boundary)
			{
				center = grid.delta*grid.L/2;
			}
			else if (spherical_boundary)
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

		// set epigenetic sequence
		if (load_chipseq) {
			int marktype = 0;
			for (std::string chipseq_file : chipseq_files)
			{
				std::ifstream IFCHIPSEQ;
				int tail_marked;
				IFCHIPSEQ.open(chipseq_file);
				for(int i=0; i<nbeads; i++)
				{
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

				// assigns methylation marks based on chipseq data 
				/*
				for(int i=0; i<nbeads; i++) {
					IFCHIPSEQ >> ntails_methylated;
					if (ntails_methylated == 2) {
						beads[i].tail1.mark = 1;
						beads[i].tail2.mark = 1;
						beads[i].tails_methylated = 2;
					}
					else if (ntails_methylated == 1) {
						beads[i].tail1.mark = 1;
						beads[i].tails_methylated = 1;
					}
				}
				*/
			}
		}
		else {
			// assigns methylation marks in a block polymer manner
			int blocksize  = nbeads/5;
			for(int i=0; i<nbeads; i++) {
				if (i%blocksize < blocksize/3) {
					beads[i].tail1.mark = 1;
					beads[i].tail2.mark = 1;
					beads[i].tails_methylated = 2;
				}
				else if (i%blocksize < 2*blocksize/3) {
					beads[i].tail1.mark = 1;
					beads[i].tails_methylated = 1;
				}
			}
		}

                // set domain sequence
		if (AB_block) {
			for(int i=0; i<nbeads; i++)
			{
				//if (i%domainsize < domainsize/2.0)
				//{
				//	beads[i].d = Bead::A;
				//}
				//else
				//{
				//	beads[i].d = Bead::B;
				//}

				double sig = 500.0;
				double prob = gaussian(2000.0, sig, i) + gaussian(6000.0, sig, i);
				double rndDouble = (double)rand() / RAND_MAX;

				if(rndDouble < prob)
				{
					beads[i].d[0] = 1; //Bead::A;
				}
				else
				{
					beads[i].d[1]= 1; // Bead::B;
				}
			}
		}

		// set bonds
		bonds.resize(nbeads-1); // use default constructor
		for(int i=0; i<nbeads-1; i++)
		{
			bonds[i] = new DSS_Bond{&beads[i], &beads[i+1]};
		}

		// output for vis.
		dumpData();
		//dumpEnergy(0);
		std::cout << "Objects created" << std::endl;
	}

	double gaussian(double mu, double sig, double x)
	{
		// un-normalized gaussian
		return std::exp(-std::pow(x-mu,2)/(2*sig*sig));
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

		double U = grid.energy(flagged_cells, chis);

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
		//std::cout << "NonBonded took " << duration.count() << "microseseconds "<< std::endl;
		return U;
	}

	double getBindingEnergy(const std::unordered_set<Cell*>& flagged_cells)
	{
		// JUST CALCULATES THE VOLUME INTERACTION OF HP1, NOT ENERGY OF SPECIFIC TAIL BINDING
		auto start = std::chrono::high_resolution_clock::now();
		
		double Energy = grid.tailEnergy(flagged_cells);

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
		//std::cout << "Binding took " << duration.count() << "microseseconds "<< std::endl;
	
		return Energy; 
	}

	double getTotalEnergy(int first, int last, const std::unordered_set<Cell*>& flagged_cells)
	{
		double U = 0;
		if (bonded_on) U += getBondedEnergy(first, last);
		if (nonbonded_on) U += getNonBondedEnergy(flagged_cells);
		if (binding_on) U += getBindingEnergy(flagged_cells);
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

			Timer t_translation("translating", print_MC);
			for(int j=0; j<n_trans; j++)
			{
				MCmove_translate();
			}
			t_translation.~Timer();


			Timer t_displace("displacing", print_MC);
			for(int j=0; j<n_disp; j++)
			{
				MCmove_displace();
			}
			t_displace.~Timer();


			Timer t_crankshaft("Cranking", print_MC);
			for(int j=0; j<n_crank; j++) {
				MCmove_crankshaft();
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
			}
			t_pivot.~Timer();


			Timer t_bind("Binding", print_MC);
			for(int j=0; j<n_bind; j++) {
				MCmove_bind();
			}
			t_bind.~Timer();

			if (production)
			{
				// note all contacts every sweep
				updateContacts();
				dumpObservables(sweep);
			}

			if (sweep%dump_frequency == 0) {
				std::cout << "Sweep number " << sweep << std::endl;
				dumpData();
				
				if (print_acceptance_rates) {
					std::cout << "acceptance rate: " << (float) acc/((sweep+1)*nSteps)*100.0 << "%" << std::endl;
					std::cout << "trans: " << (float) acc_trans/((sweep+1)*n_trans)*100 << "% \t";
					std::cout << "crank: " << (float) acc_crank/((sweep+1)*n_crank)*100 << "% \t";
					std::cout << "pivot: " << (float) acc_pivot/((sweep+1)*n_pivot)*100 << "% \t";
					std::cout << "bind: " << (float) acc_bind/((sweep+1)*n_bind)*100 << "% \t";
					std::cout << "rot: " << (float) acc_rot/((sweep+1)*n_rot)*100 << "%" << std::endl;;
				}
			}

			if (sweep%dump_stats_frequency == 0)
			{
				if (production) {dumpContacts();}
				//if (production) {dumpObservables(sweep);}

				Timer t_allenergy("all energy", print_MC);
				double bonded = getAllBondedEnergy();

				double nonbonded = 0;
				double binding = 0;
				if (nonbonded_on) nonbonded = getNonBondedEnergy(grid.active_cells);
				if (binding_on) binding = binding_on ? getBindingEnergy(grid.active_cells) : 0;
				//std::cout << "binding " << bonded << " nonbonded " << nonbonded << " binding" << binding <<  std::endl;
				t_allenergy.~Timer();

				dumpEnergy(sweep, bonded, nonbonded, binding);
			}

		}

		std::cout << "acceptance rate: " << (float) acc/(nSweeps*nSteps)*100.0 << "%" << std::endl;
	}

	
	void MCmove_displace()
	{
		Timer t_displacemove("Displacement move");
		// pick random particle
		int o = floor(beads.size()*rng->uniform());

		// copy old info (don't forget orientation, etc)
		Cell* old_cell = grid.getCell(beads[o]);
		
		Eigen::RowVector3d displacement;
		displacement = step_disp*unit_vec(displacement);

		Eigen::RowVector3d new_location = beads[o].r + displacement;
		Cell* new_cell = grid.getCell(new_location);

		// check if exited the simulation box, if so reject the move
		if (outside_boundary(new_location))
		{
			return;
		}

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
		Timer t_setup("setup", print_trans);

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

		t_setup.~Timer();

		// execute move
		try
		{
			Timer t_bounds("bounds", print_trans);
			// reject immediately if moved out of simulation box, no cleanup necessary
			for(int i=first; i<=last; i++)
			{
				new_loc = beads[i].r + displacement;
				if (outside_boundary(new_loc)) {
					throw "exited simulation box";	
				}
			}
			t_bounds.~Timer();

			Timer t_flag("Flag cells", print_trans);
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
			t_flag.~Timer();

			Timer t_uold("Uold", print_trans);
			//std::cout << "Beads: " << last-first << " Cells: " << flagged_cells.size() << std::endl;
			double Uold = getTotalEnergy(first, last, flagged_cells);
			t_uold.~Timer();

			Timer t_disp("Displacement", print_trans);
			for(int i=first; i<=last; i++)
			{
				beads[i].r += displacement;
			}
			t_disp.~Timer();

			Timer t_swap("Bead Swaps", print_trans);
			// update grid <bead index,   <old cell , new cell>>
			//for(std::pair<int, std::pair<Cell*, Cell*>> &x : bead_swaps)
			for(auto const &x : bead_swaps)
			{
				x.second.first->moveOut(&beads[x.first]);
				x.second.second->moveIn(&beads[x.first]);
			}
			t_swap.~Timer();

			Timer t_unew("Unew", print_trans);
			double Unew = getTotalEnergy(first, last, flagged_cells);
			t_unew.~Timer();

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
			if (binding_on) Uold += getBindingEnergy(flagged_cells);

			
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
			if(binding_on) Uold += getBindingEnergy(flagged_cells);
			
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

	void MCmove_bind()
	{
		// pick random particle
		int o = floor(beads.size()*rng->uniform());

		bool flip2 = round(rng->uniform());       // flip 1 or 2 tails?
		bool flip_first = round(rng->uniform());  // if 1, which tail to flip? 

		Cell* current_cell = grid.getCell(beads[o]); 
		double Uold = beads[o].tailEnergy(getChemPot()) + current_cell->getTailEnergy();  

		if (flip2) {
			beads[o].tail1.flipBind(current_cell);
			beads[o].tail2.flipBind(current_cell);
		}
		else {  // only flip one tail
			(flip_first) ? beads[o].tail1.flipBind(current_cell) : beads[o].tail2.flipBind(current_cell);
		}

		double Unew = beads[o].tailEnergy(getChemPot()) + current_cell->getTailEnergy();

		if (rng->uniform() < exp(Uold-Unew))
		{
			//std::cout << "Accepted"<< std::endl;
			acc += 1;
			acc_bind += 1;
		}
		else
		{
			//std::cout << "Rejected" << std::endl;
			
			if (flip2) {
				beads[o].tail1.flipBind(current_cell);
				beads[o].tail2.flipBind(current_cell);
			}
			else {
				(flip_first) ? beads[o].tail1.flipBind(current_cell) : beads[o].tail2.flipBind(current_cell);
			}
		}
	}
	
	void printTailComposition()
	{
		double HP1_un = 0;
		double HP1_on = 0;
		double HP1_di = 0;

		double HP1_un_tot = 0;
		double HP1_on_tot = 0;
		double HP1_di_tot = 0;

		if (!load_chipseq)
		{
			HP1_un_tot = nbeads/3*2;
			HP1_on_tot = nbeads/3*2;
			HP1_di_tot = nbeads/3*2;
		}
		
		for(Bead bead : beads)
		{
			if (bead.tails_methylated == 0) {
				HP1_un += bead.nbound();
			}
			else if (bead.tails_methylated == 1) {
				HP1_on += bead.nbound();
			}
			else { HP1_di += bead.nbound();}
		}

		//std::cout << "free HP1 is : " << HP1_free << std::endl; // need to modify for Sim::HP1_free
		std::cout << "none: " << HP1_un/HP1_un_tot << " me3: " << HP1_on/HP1_on_tot << " 2xme3: " << HP1_di/HP1_di_tot << std::endl;
	}

	void dumpData() 
	{ 
		xyz_out = fopen("./data_out/output.xyz", "a");
		fprintf(xyz_out, "%d\n", nbeads);
		fprintf(xyz_out, "atoms\n");

		for(Bead bead : beads)
		{
		    //int epimark = bead.tail1.mark + bead.tail2.mark;
		    //int bindmark = bead.tail1.bound + bead.tail2.bound;
		    //fprintf(xyz_out, "%d\t %lf\t %lf\t %lf\t %d\t %d\t %d\n" , bead.id,bead.r(0),bead.r(1),bead.r(2), epimark, bead.nbound(), bead.d);


			fprintf(xyz_out, "%d\t %lf\t %lf\t %lf\t" , bead.id,bead.r(0),bead.r(1),bead.r(2));

			for(int i=0; i<nspecies; i++)
			{
				fprintf(xyz_out, "%d\t", bead.d[i]);
			}

			fprintf(xyz_out, "\n");
		}
		fclose(xyz_out); 
	}

	void dumpEnergy(int sweep, double bonded=0, double nonbonded=0, double binding=0)
	{
		energy_out = fopen("./data_out/energy.traj", "a");
		fprintf(energy_out, "%d\t %lf\t %lf\t %lf\n", sweep, bonded, nonbonded, binding);
		fclose(energy_out);
	}

	void dumpObservables(int sweep)
	{
		//unsigned long n_AA = grid.get_AA_Contacts();
		//unsigned long n_AB = grid.get_AB_Contacts();
		//unsigned long n_BB = grid.get_BB_Contacts();

		obs_out = fopen("./data_out/observables.traj", "a");
		//fprintf(obs_out, "%d\t %ld\t %ld\t %ld\n", sweep, n_AA, n_AB, n_BB);
		fprintf(obs_out, "%d\t", sweep);

		for (int i=0; i<nspecies; i++)
		{
			for (int j=i; j<nspecies; j++)
			{
				unsigned long ij_contacts = grid.get_ij_Contacts(i, j);
				fprintf(obs_out, "%ld\t", ij_contacts);
			}
		}

		fprintf(obs_out, "\n");
		fclose(obs_out);
	}

	void dumpContacts()
	{
		// overwrites contact file with most current values
		std::ofstream contactsOutFile("./data_out/contacts.txt");
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
		initialize(); // simulation
		grid.generate();  // creates the grid locations
		grid.meshBeads(beads);  // populates the grid locations with beads;
		grid.setActiveCells();  // populates the active cell locations
		setupContacts();
		MC();
		dumpContacts();
		assert (grid.checkCellConsistency(nbeads));
		assert (grid.checkHp1Consistency(hp1_total));
	}
};

int Sim::hp1_free;

void Tail::flipBind(Cell* cell_inside)
{
	// update HP1 globally and locally in the cell
	if (bound)
	{
		cell_inside->local_HP1 -= 1;
		Sim::hp1_free += 1;
	}
	else {
		cell_inside->local_HP1 += 1;
		Sim::hp1_free -= 1;
	}

	// flip the binding state
	bound = !bound;
}

bool Grid::checkHp1Consistency(int hp1_tot)
{
	// checks to see if the number of HP1 proteins is conserved
	double cellhp1 = 0;
	for(Cell* cell : active_cells)
	{
		cellhp1 += cell->local_HP1;
	}

	cellhp1 += Sim::hp1_free;
	return (cellhp1 == hp1_tot);
}

int main()
{
	auto start = std::chrono::high_resolution_clock::now();

	Sim mySim;
	mySim.run();

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
	std::cout << "Took " << duration.count() << "seconds "<< std::endl;
	std::cout << "Moved " << nbeads_moved << " beads " << std::endl;
	return 0;
}
