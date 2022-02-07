#pragma once

#include <iostream>
#include <vector>
#include <unordered_set>

#include "Eigen/Dense"
#include "Bead.h"
#include "Cell.h"


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

	void generate();
	void setActiveCells();
	void printActiveCells();
	void meshBeads(std::vector<Bead> &beads);
	Cell* getCell(const Bead& bead);
	Cell* getCell(const Eigen::RowVector3d &r);
	bool checkCellConsistency(int nbeads);
	double densityCapEnergy(const std::unordered_set<Cell*>& flagged_cells);
	double energy(const std::unordered_set<Cell*>& flagged_cells, const Eigen::MatrixXd &chis);
	double diagEnergy(const std::unordered_set<Cell*>& flagged_cells, const std::vector<double> diag_chis);
	double boundaryEnergy(const std::unordered_set<Cell*>& flagged_cells, const double boundary_chi);
	double boundaryEnergy(const std::unordered_set<Cell*>& flagged_cells, const std::vector<std::vector<double>> &Smatrix);
	double SmatrixEnergy(const std::unordered_set<Cell*>& flagged_cells, const std::vector<std::vector<double>> &Smatrix, const Eigen::MatrixXd &chis);
	double EmatrixEnergy(const std::unordered_set<Cell*>& flagged_cells, const std::vector<std::vector<double>> &Smatrix, const Eigen::MatrixXd &chis);
	double get_ij_Contacts(int i, int j) ;
	void getDiagObs(std::vector<double> &diag_obs);
	double cellCount();
	double getChromatinVolfrac();
	double getChromatinVolfrac2();
	double getChromatinVolfracD();
};
