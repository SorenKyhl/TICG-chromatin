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
#include "/home/coraor/cnpy/cnpy.h"
#include "main.cpp"

#include "Eigen/Dense"

class Tail;
class Bead;



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

// abstract class
class Angle {
public:
	Angle(Bead* b1, Bead* b2, Bead* b3)
		: pbead1{b1}, pbead2{b2}, pbead3{b3} {}

	Bead* pbead1;
	Bead* pbead2;
	Bead* pbead3;

	void print() {
		std::cout << pbead1->id << " " << pbead2->id << " " << pbead3->id << std::endl;
	}

	virtual double energy() = 0;
}

// abstract class
class Dihedral {
public:
	Dihedral(Bead* b1, Bead* b2, Bead* b3, Bead* b4)
		: pbead1{b1}, pbead2{b2}, pbead3{b3}, pbead4{b4} {}

	Bead* pbead1;
	Bead* pbead2;
	Bead* pbead3;
	Bead* pbead4;

	void print() {
		std::cout << pbead1->id << " " << pbead2->id << " " << pbead3->id;
		std::cout << " " << pbead4->id << std::endl;
	}

	virtual double energy() = 0;
}


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
		// DELTA = 1
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
		double delta = 16.5;       // dimless (is it 16.5 or 0.33?)
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



//Lookuptable for energy
class LookupTable3D {
public:

	int NRL;
	std::vector<std::vector<std::vector<double>>> values;
	std::vector<std::vector<double>> range;


	LookupTable3D(int nrl)
		: NRL{nrl} {

	//Setup values
	cnpy::NpyArray arr = cnpy::npy_load(
		"/project2/depablo/coraor/fiber_builder/second_cond_" + 
			std::to_string(NRL) + "_kde.dat.npy");

	std::vector<std::vector<std::vector<double>>> values = arr.data<std::vector<std::vector<std::vector<double>>>>();

	//Setup range
	std::ifstream ran("/project2/depablo/coraor/fiber_builder/ranges_" + 
			std::to_string(NRL) + "_kde.dat.npy");

	//read data out from range
	std::vector<std::vector<double>> range;
	std::string str;
	while(std::getline(ran,str))
	{
		//Line contains string of length > 0, save in vector
		if(str.size() > 0) {
			std::vector<double> line;
			std::istringstream s(str);
			std::copy(std::istream_iterator<double>(s)
			, std::istream_iterator<double>()
			, std::back_inserter(line)
			);
			ranstr.push_back(line);
		}
	}
	size_t i=0;
	while (i < 5){
		std::cout << range[0][i] <<  " ";
	}

	
	}

	//Calculate the interpolated triplet energy 
	double energy(double alpha, double beta, double alphaprime)
	{
		//Calculate grid interpolation

	}
}









