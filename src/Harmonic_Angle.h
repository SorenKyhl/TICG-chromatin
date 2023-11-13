#pragma once

#include "Angle.h"
#include <cmath>

class Harmonic_Angle : public Angle {
public:
	Harmonic_Angle(Bead* bead1, Bead* bead2, Bead* bead3, double kk, double t00)
		: Angle{bead1, bead2, bead3}, k{kk}, t0{t00} {}
	// cosine angle potential
	// t0 is in degrees

	double k;
	double t0;
	double t0rad = t0/180*3.14159;

	double energy()
	{
		Eigen::RowVector3d displacement1 = pbead2->r - pbead1->r;
		displacement1 /= displacement1.norm();
		if (t0 == 180){
			// define displacement2 such as to avoid cos^(-1)
			Eigen::RowVector3d displacement2 = pbead3->r - pbead2->r;
			displacement2 /= displacement2.norm();
			return k*(1-displacement1.dot(displacement2));
		} else {
			Eigen::RowVector3d displacement2 = pbead2->r - pbead3->r;
			displacement2 /= displacement2.norm();
			double cos_t = displacement1.dot(displacement2);
			double t = acos(cos_t);
			return k*(1-cos(t-t0rad));
		}
	}
};
