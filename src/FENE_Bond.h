#pragma once

#include "Bead.h"
#include "Bond.h"

class FENE_Bond : public Bond {
public:
	FENE_Bond(Bead* bead1, Bead* bead2, double kk, double r00, double bb)
		: Bond{bead1, bead2}, k{kk}, r0{r00}, b{bb} {}

	double k;
	double r0;
	double r0squared = r0 * r0;
	double b = b;
	double b216 = b * std::pow(2, 1.0 / 6.0);

	double energy()
	{
		Eigen::RowVector3d displacement = pbead2 ->r - pbead1->r;
		double r = sqrt(displacement.dot(displacement));
		double result = 0;
		// FENE
		if (r < r0){
			result += -0.5*k*r0squared*log(1-r/r0);
		}
		// HC
		if (r < b216){
			result += 4*(std::pow(b/r, 12) - std::pow(b/r, 6) + (1/4));
		}


		return result;
	}
};
