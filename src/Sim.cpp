#include "Sim.h"

// assign variable to json key of the same name
#define READ_JSON(json, var) read_json((json), (var), #var)
template<class T>
void read_json(const nlohmann::json& json, T& var, std::string varname) {
    std::cout << "loading " << varname << " ... ";
	if( !json.contains(varname) ) {
		throw std::runtime_error("config.contains(\""+varname+"\") has failed");
	}
	var = json[varname];
    std::cout << "loaded: " << std::to_string(var) << std::endl;
}

void Sim::run() {
	readInput();            // load parameters from config.json
	if (smatrix_on) { setupSmatrix(); }
	calculateParameters();  // calculates derived parameters
	makeOutputFiles();      // open files 
	initialize();           // set particle positions and construct bonds
	grid.generate();        // creates the grid locations
	grid.meshBeads(beads);  // populates the grid locations with beads;
	grid.setActiveCells();  // populates the active cell locations
	setupContacts();        // construct contact map 
	MC();                   // MC simulation
	assert (grid.checkCellConsistency(nbeads));
}

void Sim::setupContacts() {
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

void Sim::updateContacts() {
	if (update_contacts_distance){
		updateContactsDistance();
	}
	else {
		updateContactsGrid();
	}
}

void Sim::updateContactsGrid() {
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

void Sim::updateContactsDistance() {
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


Eigen::MatrixXd Sim::unit_vec(Eigen::MatrixXd b) {
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

void Sim::readInput() {
	// reads simulation parameters from config.json file

	std::cout << "reading input file ... " << std::endl;

	std::ifstream i("config.json");
	if ( !i.good() )
	{
		throw std::runtime_error("config.json does not exist or cannot be opened");
	}

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
				throw std::logic_error("Number of chipseq files: " 
						+ std::to_string(chipseq_files.size()) 
						+ " must equal number of species: " + std::to_string(nspecies));
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
	assert(config.contains("prof_timer_on")); prof_timer_on = config["prof_timer_on"];
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
	assert(config.contains("smatrix_filename")); smatrix_filename = config["smatrix_filename"];
	assert(config.contains("smatrix_on")); smatrix_on = config["smatrix_on"];
	assert(config.contains("phi_solvent_max")); Cell::phi_solvent_max = config["phi_solvent_max"];

	//cellcount_on = config["cellcount_on"];

	assert(config.contains("seed"));
	int seed = config["seed"];
	rng = new RanMars(seed);

	std::cout << "read successfully" << std::endl;
}

void Sim::makeOutputFiles() {
	std::cout << "making output files ... ";

	// make and populate data output directory
	std::string command = "mkdir " + data_out_filename;
	const int dir_err = system(command.c_str());
	if (dir_err == -1) {std::cout << "error making data_out directory" << std::endl;}

	xyz_out_filename = "./" + data_out_filename + "/output.xyz";
	energy_out_filename = "./" + data_out_filename + "/energy.traj";
	obs_out_filename = "./" + data_out_filename + "/observables.traj";
	diag_obs_out_filename = "./" + data_out_filename + "/diag_observables.traj";
	density_out_filename = "./" + data_out_filename + "/density.traj";


	// fopen(char*, char*) function signature takes c strings...
	// std::string overloads + operator and resturns std::string, need to convert back to c_str()
	xyz_out = fopen(xyz_out_filename.c_str(), "w");
	energy_out = fopen((energy_out_filename).c_str(), "w");
	obs_out = fopen((obs_out_filename).c_str(), "w");
	diag_obs_out = fopen((diag_obs_out_filename).c_str(), "w");
	density_out = fopen((density_out_filename).c_str(), "w");

	std::cout << "created successfully" << std::endl;
}

bool Sim::outside_boundary(Eigen::RowVector3d r) {
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
	
void Sim::initialize() {
	std::cout << "Initializing simulation objects ... " << std::endl;
	Timer t_init("Initializing");


	// set configuration
	beads.resize(nbeads);  // uses default constructor initialization to create nbeads;
	std::cout << "load configuratin is " << load_configuration << std::endl;
	if(load_configuration) 
	{
		loadConfiguration();
	}
	else {
		bond_length = 16.5; // nm
		initRandomCoil(bond_length);
	}

	// set up chipseq
	if (load_chipseq) { loadChipseq(); }

	// set bonds
	constructBonds();

	// output initial xyz configuration
	dumpData();
	std::cout << "Objects created" << std::endl;
}

void Sim::calculateParameters() {
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
}

void Sim::loadConfiguration() {
	// loads x,y,z positions for every particle from <inputfile>.xyz file
	
	std::cout << "Loading configuration from " << load_configuration_filename << std::endl;
	std::ifstream IFILE; 
	IFILE.open(load_configuration_filename); 
	if ( !IFILE.good() ) 
	{
		throw std::runtime_error(load_configuration_filename + " does not exist or could not be opened");
	}
	std::string line;
	getline(IFILE, line); // nbeads line
	std::cout << line << std::endl;

	int init_nbeads = stoi(line);
	std::cout << "checking if nbeads in config.json matches number of beads in the first line of <input>.xyz ... " << std::endl;
	assert(init_nbeads == nbeads);
	std::cout << "nbeads in config.json matches <input>.xyz" << std::endl;

	
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

void Sim::initRandomCoil(double bondlength) {	
	// generates x,y,z positions for all particles according to a random coil

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

void Sim::loadChipseq() {
	// set up chipseq
	int marktype = 0;
	for (std::string chipseq_file : chipseq_files)
	{
		std::ifstream IFCHIPSEQ;
		IFCHIPSEQ.open(chipseq_file);
		if ( IFCHIPSEQ.good() )
		{
			for(int i=0; i<nbeads; i++)
			{
				//beads[i].d.reserve(nspecies);
				beads[i].d.resize(nspecies);
				IFCHIPSEQ >> beads[i].d[marktype];
			}
			marktype++;
			IFCHIPSEQ.close();
		}
		else
		{
			throw std::runtime_error(chipseq_file + " does not exist or could not be opened");
		}
	}
}

void Sim::constructBonds() {
	// constructs bond objects. Particle positions must already be initialized.
	bonds.resize(nbeads-1); // use default constructor
	for(int i=0; i<nbeads-1; i++)
	{
		bonds[i] = new DSS_Bond{&beads[i], &beads[i+1]};
	}
}

void Sim::print() {
	std::cout << "simulation in : "; 
	std::cout << "With beads: " << std::endl;
	for(Bead& bb : beads) bb.print();             // use reference to avoid copies
	std::cout << "And bonds: " << std::endl;
	for(Bond* bo : bonds) bo->print();
}

double Sim::getAllBondedEnergy() {
	double U = 0;
	for(Bond* bond : bonds) {U += bond->energy();}
	return U;
}

double Sim::getBondedEnergy(int first, int last) {
	double U = 0;
	//for(Bond* bo : bonds) U += bo->energy();  // inefficient 
	if (first>0) U += bonds[first-1]->energy(); // move affects bond going into first
	if (last<(nbeads-1)) U += bonds[last]->energy();   // ... and leaving the second
	return U;
}

double Sim::getNonBondedEnergy(const std::unordered_set<Cell*>& flagged_cells) {
	// gets all the nonbonded energy
	auto start = std::chrono::high_resolution_clock::now();

	double U = grid.densityCapEnergy(flagged_cells);
	if (plaid_on)
	{
		if (smatrix_on)
		{
			U += grid.SmatrixEnergy(flagged_cells, smatrix);
		}
		else
		{
			U += grid.energy(flagged_cells, chis);
		}
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

double Sim::getJustDiagEnergy(const std::unordered_set<Cell*>& flagged_cells) {
	// for when dumping energy; 
	double U = grid.diagEnergy(flagged_cells, diag_chis); 
	return U;
}

double Sim::getJustBoundaryEnergy(const std::unordered_set<Cell*>& flagged_cells) {
	// for when dumping energy; 
	double U = grid.boundaryEnergy(flagged_cells, boundary_chi); 
	return U;
}

double Sim::getTotalEnergy(int first, int last, const std::unordered_set<Cell*>& flagged_cells) {
	double U = 0;
	if (bonded_on) U += getBondedEnergy(first, last);
	if (nonbonded_on) U += getNonBondedEnergy(flagged_cells);
	return U;
}

double Sim::randomExp(double mu, double decay) {
	// generates random number distributed according to two-sided exponential distribution
	// centered about mu, with characteristic decay length
	double cdf_y;
	do {
		cdf_y = rng->uniform();
	} while (cdf_y <= 0); // cdf_y cannot exactly equal 0, otherwise inverse cdf is -infty 


	if (cdf_y > 0.5) {
		return mu - decay*log(1 - 2*std::abs(cdf_y - 0.5)); // inverse cdf
	}
	else {
		return mu + decay*log(1 - 2*std::abs(cdf_y - 0.5)); // inverse cdf
	}
}

void Sim::MC() {
	std::cout << "Beginning Simulation" << std::endl;
	for(int sweep = 0; sweep<nSweeps; sweep++)
	{
		//std::cout << sweep << std::endl; 
		double nonbonded;
		//nonbonded = getNonBondedEnergy(grid.active_cells);
		//std::cout << "beginning sim: nonbonded: " <<  grid.active_cells.size() << std::endl;

		looping:
		Timer t_translation("translating", prof_timer_on);
		for(int j=0; j<n_trans; j++)
		{
			MCmove_translate();
			//nonbonded = getnonbondedenergy(grid.active_cells);
			//std::cout << nonbonded << std::endl;
		}
		//t_translation.~Timer();

		if (gridmove_on) MCmove_grid();
		//nonbonded = getNonBondedEnergy(grid.active_cells);
		//std::cout << nonbonded << std::endl;

		Timer t_displace("displacing", prof_timer_on);
		for(int j=0; j<n_disp; j++)
		{
			MCmove_displace();
			//nonbonded = getNonBondedEnergy(grid.active_cells);
			//std::cout << nonbonded << std::endl;
		}
		//t_displace.~Timer();


		Timer t_crankshaft("Cranking", prof_timer_on);
		for(int j=0; j<n_crank; j++) {
			MCmove_crankshaft();
			//nonbonded = getNonBondedEnergy(grid.active_cells);
			//std::cout << nonbonded << std::endl;
		}
		//t_crankshaft.~Timer();

		
		Timer t_rotation("Rotating", prof_timer_on);
		for(int j=0; j<n_rot; j++) {
			MCmove_rotate();
		}
		//t_rotation.~Timer();
		

		Timer t_pivot("pivoting", prof_timer_on);
		for(int j=0; j<n_pivot; j++) {
			MCmove_pivot(sweep);
			//nonbonded = getNonBondedEnergy(grid.active_cells);
			//std::cout << nonbonded << std::endl;
		}
		//t_pivot.~Timer();


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

			Timer t_allenergy("all energy", prof_timer_on);

			double bonded = getAllBondedEnergy();
			double nonbonded = 0;
			nonbonded = nonbonded_on ? getNonBondedEnergy(grid.active_cells) : 0; // includes diagonal and boundary energy
			double diagonal = 0;
			diagonal = diagonal_on ? getJustDiagEnergy(grid.active_cells) : 0;
			double boundary = 9;
			boundary = boundary_attract_on ? getJustBoundaryEnergy(grid.active_cells) : 0;
			//std::cout << "bonded " << bonded << " nonbonded " << nonbonded << std::endl;
			//t_allenergy.~Timer();

			dumpEnergy(sweep, bonded, nonbonded, diagonal, boundary);
		}

	}

	// final contact map
	dumpContacts(nSweeps);
	std::cout << "acceptance rate: " << (float) acc/(nSweeps*nSteps)*100.0 << "%" << std::endl;
}


void Sim::MCmove_displace() {
	Timer t_displacemove("Displacement move", prof_timer_on);
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

void Sim::MCmove_translate() {
	if (print_trans) std::cout << "==================NEW MOVE ====================" << std::endl;
	Timer t_trans("Translate", print_trans);
	//Timer t_setup("setup", print_trans);

	// select a first bead at random
	int first = floor(beads.size()*rng->uniform());

	// choose second bead from two-sided exponential distribution around first
	int last = -1; 
	while (last < 0 || last >= nbeads)
	{
		last = std::round(randomExp(first, exp_decay));            // does this obey detailed balance?
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

void Sim::MCmove_crankshaft() {
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
		last = std::round(randomExp(first, exp_decay_crank));
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

void Sim::MCmove_rotate() {
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

void Sim::MCmove_pivot(int sweep) {
	// index terminology:
	// pivot === the bead being pivoted around, but not itself moved
	// [first,last] === the interval of beads physically moved in the pivot
		// depending on which end the pivot is executed, this is either [0, pivot-1] or [pivot+1, nbeads-1]
	// The only bonds affected are pivot-1 OR pivot;

	// chose one end of the polymer and a pivot bead
	int end = (nbeads-1)*std::round(rng->uniform()); //  either first bead or last bead

	end = nbeads-1;

	// pick second bead according to single-sided exponential distribution away from end
	int length;
	do {
		length = std::abs(std::round(randomExp(0, exp_decay_pivot))); // length down the polymer from the end
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

void Sim::MCmove_grid() {
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


void Sim::dumpData()  { 
	xyz_out = fopen(xyz_out_filename.c_str(), "a");
	fprintf(xyz_out, "%d\n", nbeads);
	fprintf(xyz_out, "atoms\n");

	for(Bead bead : beads)
	{
		fprintf(xyz_out, "%d\t %lf\t %lf\t %lf\t" , bead.id,bead.r(0),bead.r(1),bead.r(2));

		if (plaid_on)
		{
			for(int i=0; i<nspecies; i++)
			{
				fprintf(xyz_out, "%f\t", bead.d[i]);
			}
		}

		fprintf(xyz_out, "\n");
	}
	fclose(xyz_out); 
}

void Sim::dumpEnergy(int sweep, double bonded=0, double nonbonded=0, double diagonal=0, double boundary=0) {
	energy_out = fopen(energy_out_filename.c_str(), "a");
	fprintf(energy_out, "%d\t %lf\t %lf\t %lf\t %lf\n", sweep, bonded, nonbonded, diagonal, boundary);
	fclose(energy_out);
}

void Sim::dumpObservables(int sweep) {
	if (plaid_on)
	{
		obs_out = fopen(obs_out_filename.c_str(), "a");
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
		diag_obs_out = fopen(diag_obs_out_filename.c_str(), "a");
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
		density_out = fopen(density_out_filename.c_str(), "a");
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

void Sim::dumpContacts(int sweep) {

	if (track_contactmap){
		// outputs new contact map, doesn't override
		contact_map_filename = "./" + data_out_filename + "/contacts" + std::to_string(sweep) + ".txt";
	}
	else {
		// overwrites contact file with most current values
		contact_map_filename = "./" + data_out_filename + "/contacts.txt";
	}

	std::ofstream contactsOutFile(contact_map_filename);
	for (const auto &row : contact_map) {
		for (const int &element : row) {
			contactsOutFile << element << " ";
		}
		contactsOutFile << "\n";
	}
}

void Sim::setupSmatrix() {
	std::ifstream smatrixfile(smatrix_filename);

	if ( !smatrixfile.good() ) {
		throw std::runtime_error(smatrix_filename + " does not exist or cannot be opened");
	}

	smatrix.resize(nbeads);
	for (int i=0; i<nbeads; i++) {
		smatrix[i].resize(nbeads);
		for (int j=0; j<nbeads; j++) {
			smatrixfile >> smatrix[i][j];
		}
	}
	std::cout << "loaded Smatrix, first element:" << smatrix[0][0] << std::endl;
}
