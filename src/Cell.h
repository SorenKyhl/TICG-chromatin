#pragma once 

#include <iostream>
#include <vector>
#include <unordered_set>
#include "Eigen/Dense"
#include "Bead.h"


class Cell {
public:
	Eigen::RowVector3d r; // corner of cell RELATIVE TO ORIGIN... the grid origin diffuses
	std::unordered_set<Bead*> contains; // beads associated inside this gridpoint
	double vol;		                // volume of cell
	static const int beadvol = 520; // volume of a bead in the cell. 

	static int ntypes;  // number of bead types
	std::vector<double> typenums = std::vector<double>(ntypes); // always up-to-date
	std::vector<double> phis = std::vector<double>(ntypes); // only up-to-date after energy calculation
	
	static double diag_binsize;
	static int diag_nbins;
	std::vector<double> diag_phis = std::vector<double>(diag_nbins);
	static bool diagonal_linear;

	static double phi_solvent_max;
	static double phi_chromatin;
	static double kappa;
	static bool density_cap_on;
	static bool compressibility_on;

	void print();
	void reset();
	void moveIn(Bead* bead);
	void moveOut(Bead* bead);
	double getDensityCapEnergy();
	double getEnergy(const Eigen::MatrixXd &chis);
	double getDiagEnergy(const std::vector<double> diag_chis);
	double getBoundaryEnergy(const double boundary_chi, const double delta);
	double getSmatrixEnergy(const std::vector<std::vector<double>> &Smatrix);

};


