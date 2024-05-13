#pragma once

#include "Bead.h"
#include "Bond.h"

class SC_Bond : public Bond {
public:
	SC_Bond(Bead* bead1, Bead* bead2, double kk, double r00)
		: Bond{bead1, bead2}, k{kk}, r0{r00} {}

	double k;
	double r0;

	double energy()
	{
		Eigen::RowVector3d displacement = pbead2 ->r - pbead1->r;
		double r = sqrt(displacement.dot(displacement));

		double diff = r-r0;
		double diffSquared = std::pow(diff, 2);
		double diffCubed = diffSquared * diff;
		double diffFourth = diffCubed * diff;

		double result = 0;
		result += diffSquared;
		result += diffCubed;
		result += diffFourth;
		result *= k;

		return result;
	}
};
