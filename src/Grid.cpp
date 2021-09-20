
#include "Grid.h"

void Grid::generate() {
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
};

void Grid::setActiveCells() {
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
};

void Grid::printActiveCells() {
	// prints only the active cells that contains particles
	std::cout << "Printing Active Cells: number:" << active_cells.size() << std::endl;
	for(Cell* cell : active_cells) {
		if(cell->contains.size() > 0) cell->print();
	}
};

void Grid::meshBeads(std::vector<Bead> &beads) {
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
};

Cell* Grid::getCell(const Bead& bead) {
	// Returns a pointer to the cell in which a bead is located
	int i, j, k;
	i = floor((bead.r(0) - origin(0))/delta);
	j = floor((bead.r(1) - origin(1))/delta);
	k = floor((bead.r(2) - origin(2))/delta);
	return &cells[i][j][k];
};

Cell* Grid::getCell(const Eigen::RowVector3d &r) {
	// Returns a pointer to the cell in which a bead is located
	int i, j, k;
	i = floor((r(0) - origin(0))/delta);
	j = floor((r(1) - origin(1))/delta);
	k = floor((r(2) - origin(2))/delta);
	return &cells[i][j][k];
};

bool Grid::checkCellConsistency(int nbeads) {
	// checks to see if number of beads in the grid is the same as simulation nbeads
	double cellbeads = 0;
	for(Cell* cell : active_cells)
	{
		cellbeads += cell->contains.size();
	}

	return (cellbeads == nbeads); 
};

double Grid::densityCapEnergy(const std::unordered_set<Cell*>& flagged_cells) {
	// energy penalty due to density cap
	double U = 0; 
	for(Cell* cell : flagged_cells)
	{
		U += cell->getDensityCapEnergy();
	}
	return U;
};
		
double Grid::energy(const std::unordered_set<Cell*>& flagged_cells, const Eigen::MatrixXd &chis) {
	// nonbonded volume interactions
	double U = 0; 
	for(Cell* cell : flagged_cells)
	{
		U += cell->getEnergy(chis);
	}
	return U;
};

double Grid::diagEnergy(const std::unordered_set<Cell*>& flagged_cells, const std::vector<double> diag_chis) {
	// nonbonded volume interactions
	double U = 0; 
	for(Cell* cell : flagged_cells)
	{
		U += cell->getDiagEnergy(diag_chis);
	}
	return U;
};

double Grid::boundaryEnergy(const std::unordered_set<Cell*>& flagged_cells, const double boundary_chi) {
	// nonbonded volume interactions
	double U = 0; 
	for(Cell* cell : flagged_cells)
	{
		U += cell->getBoundaryEnergy(boundary_chi, delta);
	}
	return U;
};

double Grid::SmatrixEnergy(const std::unordered_set<Cell*>& flagged_cells, const std::vector<std::vector<double>> &Smatrix) {
	// nonbonded volume interactions
	double U = 0; 
	for(Cell* cell : flagged_cells)
	{
		U += cell->getSmatrixEnergy(Smatrix);
	}
	return U;
};

double Grid::get_ij_Contacts(int i, int j) 
{
	// calculates average phi_i phi_j
	double obs  = 0;
	for(Cell* cell : active_cells)
	{
		obs += cell->phis[i] * cell->phis[j];
	}

	obs /= active_cells.size(); 
	return obs;
};

void Grid::getDiagObs(std::vector<double> &diag_obs) {
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
};

double Grid::cellCount() {
	int sum = 0;
	int num = 0;
	for (Cell* cell : active_cells)
	{
		sum += cell->contains.size();
		num++;
	}

	return (double) sum / (double) num;
};
