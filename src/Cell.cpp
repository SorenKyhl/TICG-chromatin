#include "Cell.h"
#include "cmath"

//TOOD: phase out
double Cell::beadvol;
int Cell::ntypes;
int Cell::diag_nbins;
int Cell::diag_binsize;
bool Cell::diagonal_linear; //TODO this is deprecated
double Cell::phi_solvent_max;
double Cell::phi_chromatin;
double Cell::kappa;
bool Cell::density_cap_on;
bool Cell::compressibility_on;
bool Cell::double_count_main_diagonal;
bool Cell::dense_diagonal_on;
int Cell::n_small_bins;
int Cell::n_big_bins;
int Cell::small_binsize;
int Cell::big_binsize;
int Cell::diag_start;

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
			// phi_solvent_max is the minimum solvent volume fraction (Defaults to 0.5)
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

double Cell::getConstantEnergy(const double constant_chi) {
	// constant nonbonded interaction between all pairs of beads
	double U = constant_chi*pow(contains.size(), 2)*beadvol/vol;

	return U;
};

double Cell::getSLmatrixEnergy(const Eigen::MatrixXd &SLmatrix)
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
		for(int j=i; j<imax; j++)
		{

			U += SLmatrix(indices[i],indices[j]) * beadvol/vol;
		}
	}
	return U;
}

double Cell::getDmatrixEnergy(const Eigen::MatrixXd &Dmatrix)
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
		for(int j=i; j<imax; j++)
		{
			U += Dmatrix(indices[i], indices[j]) * beadvol/vol * 2;
		}
	}
	return U;
}

int Cell::binDiagonal(int d)
{
	int bin_index = -1;

	if (Cell::dense_diagonal_on)
	{
		int dense_cutoff = Cell::n_small_bins * Cell::small_binsize;
		// diagonal chis are binned in a dense set (small bins) from d=0 to d=dense_cutoff,
		// then a sparse set (large bins) from d=cutoff to d=len(hic)
		if ( d == 0 )
		{
			// this clause avoids divide by zero error
			bin_index = 0;
		}
		else if ( d > dense_cutoff )
		{
			bin_index = Cell::n_small_bins + std::floor( (d - dense_cutoff) / Cell::big_binsize );
		}
		else
		{
			bin_index = std::floor( d / Cell::small_binsize );
		}
	}
	else
	{
		// diagonal chis are linearly spaced from d=0 to d=nbeads
		bin_index = std::floor( d / Cell::diag_binsize );
	}
	return bin_index;
}

double Cell::getDiagEnergy(const std::vector<double> diag_chis) {
    for (int i = 0; i < diag_nbins; i++) {
        diag_phis[i] = 0;
    }

    int d_index; // genomic separation (index for diag_phis)
    int imax = (int)contains.size();
    std::vector<int> indices;
    for (const auto &elem : contains) {
        indices.push_back(elem->id);
    }

    // count pairwise contacts
    for (int i = 0; i < imax; i++) {
        for (int j = i; j < imax; j++) {
            int d = std::abs(indices[i] - indices[j]);
            if (d >= diag_start) {
                d_index = binDiagonal(d);
                int nbonds;
                if (Cell::double_count_main_diagonal)
                {
                    nbonds = 2;  // both main and off diagonal count twice
                }
                else {
                    nbonds = d ? 2 : 1;       // count two for all off-diagonal
                }
                diag_phis[d_index] += nbonds; // diag phis is just a count,
                                              // multiply by volumes later
            }
        }
    }

    double Udiag = 0;
    for (int i = 0; i < diag_nbins; i++) {
        Udiag += diag_chis[i] * diag_phis[i];
    }
    return Udiag * beadvol / vol;
};

double Cell::bonds_to_beads(int bonds)
{
	// convert number of bonds to number of pseudobeads
	// correct way:
	// include self-self interactions
	// bonds = beads**2/2
	//
	// wrong way:
	// bonds = beads*(beads-1)/2
	// solving for beads, using quadratic formula:
	// beads = (1 + sqrt(1 + 8*bonds)) / 2
	// return  (1 + sqrt( 1 + 8*bonds)) / 2.0 ;
	return sqrt(2*bonds);
}

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
