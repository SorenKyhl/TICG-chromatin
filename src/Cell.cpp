#include "Cell.h"

//TOOD: phase out
int Cell::ntypes;
int Cell::diag_nbins;
double Cell::diag_binsize;
bool Cell::diagonal_linear;
double Cell::phi_solvent_max;
double Cell::phi_chromatin;
double Cell::kappa;
bool Cell::density_cap_on;
bool Cell::compressibility_on;

void Cell::print() {
	std::cout << r << "     N: " << contains.size() << std::endl;
	for (Bead* bead : contains)
	{
		bead->print();
	};
};

void Cell::reset() {
	// clears population trackers
	contains.clear();
	std::fill(typenums.begin(), typenums.end(), 0);  // DO NOT USE .clear()
	std::fill(phis.begin(), phis.end(), 0);      // ... it doesn't re-assign to 0's
};

void Cell::moveIn(Bead* bead) {
	// updates local number of each type of bead, but does not recalculate phis
	// TODO update populations of distance ids
	contains.insert(bead);
	for(int i=0; i<ntypes; i++)
	{
		typenums[i] += bead->d[i];
	}
};

void Cell::moveOut(Bead* bead) {
	// updates local number of each type of bead, but does not recalculate phis 
	// TODO update populations of distance ids
	contains.erase(bead);
	for(int i=0; i<ntypes; i++)
	{
		typenums[i] -= bead->d[i];
	}
};

double Cell::getDensityCapEnergy() {
	// Density in each cell is capped at phi_solvent_max
	// otherwise, incur a large energy penalty
	
	float phi_beads = contains.size()*beadvol/vol;
	float phi_solvent = 1 - contains.size()*beadvol/vol;

	double U = 0;
	if (density_cap_on)
	{
		if (phi_solvent < phi_solvent_max)
		{
			// high volume fraction occurs when more than 50% of the volume is occupied by beads
			U = 99999999999*phi_beads;
		}
	}
	else if (compressibility_on)
	{
		U = (phi_beads - phi_chromatin)*(phi_beads - phi_chromatin)*kappa;
	}
	return U;
};

double Cell::getEnergy(const Eigen::MatrixXd &chis) {
	for (int i=0; i<ntypes; i++)
	{
		phis[i] = typenums[i]*beadvol/vol;
	}

	double U = 0;
	for (int i=0; i<ntypes; i++)
	{
		for (int j=i; j<ntypes; j++)
		{
			U += chis(i,j)*phis[i]*phis[j]*vol/beadvol;
		}
	}
	return U;
};

double Cell::getSmatrixEnergy(const std::vector<std::vector<double>> &Smatrix)
{
	double U = 0;

	std::vector<int> indices;
	int imax = (int) contains.size();
	for (const auto& elem : contains)
	{
		indices.push_back(elem->id);
	}

	assert(imax == indices.size());

	for (int i=0; i<imax; i++)
	{
		for(int j=0; j<imax; j++)
		{

			U += Smatrix[indices[i]][indices[j]] * beadvol/vol;
		}
	}
	return U;
}

double Cell::getDiagEnergy(const std::vector<double> diag_chis) {
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
			//index = std::floor( std::abs(bead1->id - bead2->id) / diag_binsize);
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
			d_index  = std::floor( std::abs(indices[i] - indices[j]) / diag_binsize);
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
};

double Cell::getBoundaryEnergy(const double boundary_chi, const double delta) {
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
};
