#define _GLIBCXX_USE_CXX11_ABI 0

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

#include "Eigen/Dense"
#include "random_mars.cpp"
#include "nlohmann/json.hpp"
#include "prof_timer.cpp"
#include "/home/coraor/cnpy/cnpy.h"
#include "/home/coraor/cnpy/cnpy.cpp"
//#include "fiber.h"

unsigned long nbeads_moved = 0;
//RanMars rng(1);

class Cell; 
class LookupTable3D;
class LookupTable2D;
class LookupTable1D;


class Tail {
public:
	Tail(bool m)
		: mark{m}, bound{0} {}
	
	static constexpr double eps_methylated = -0.01;   // kT
	static constexpr double eps_unmethylated = 1.52;  // kT

	bool mark; // methylation state: me3?
	bool bound; // bound by HP1?

	double energy(double chem)
	{
		// calculates binding energy as a function of chemical potential (chem = kT log[HP1_free])
		return bound*(mark*(eps_methylated-chem) + !mark*(eps_unmethylated-chem));
	}

	void flipBind(Cell* cell_inside);

	void print()
	{
		std::cout <<  "Tail with epigenetic state: " << mark << " binding state " << bound << std::endl;
	}
};

class Bead {
public:
	Bead(int i, double x=0, double y=0, double z=0, bool mark1=0, bool mark2=0)
		: id{i}, r{x,y,z}, tail1{mark1}, tail2{mark2} {}
	
	Bead() 
		: id{0}, r{0,0,0}, tail1{0}, tail2{0} {}

	static int ntypes;
	std::vector<int> d = std::vector<int>(ntypes);

	int id;   // unique identifier: uniqueness not strictly enforced.
	Eigen::RowVector3d r; // position
	Eigen::RowVector3d u; // orientation
	bool lh_bound; // bound by LH

	Tail tail1;
	Tail tail2; 
	int tails_methylated;

	static constexpr double Jprime = -4; // kT
	static constexpr double Jlh = -2; //kT, LH binding energy

	double tailEnergy(double chem)
	{
		//    intranucleosome HP1 interaction |        HP1 binding energy
		return Jprime*tail1.bound*tail2.bound + tail1.energy(chem) + tail2.energy(chem);
	}

	double LHEnergy()
	{
		return Jlh*lh_bound;
	}

	void flipLH();

	int nbound()
	{
		// number of bound HP1
		return tail1.bound + tail2.bound;
	}

	void print() {std::cout << id <<" "<< r << std::endl;}
};


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
};

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
};


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

	double alpha_min, alpha_max, beta_min, beta_max, alphaprime_min,alphaprime_max;
	double alpha_int, beta_int, alphaprime_int;

	cnpy::NpyArray val_arr;
	
	size_t length = 200;

	LookupTable3D() {
		//std::cout << "Warning! Empty 3D Lookuptable generated!" << std::endl;
	}


	LookupTable3D(int const & nrl)
		: NRL{nrl} {


		//std::string fname = "/home/coraor/TICG-chromatin/range.npy";
		std::string fname = "/project2/depablo/coraor/fiber_builder/second_cond_" + 
				std::to_string(nrl) + "_kde.dat.npy";
		//Setup values
		//std::cout << fname;
		cnpy::NpyArray val_arr = cnpy::npy_load(fname);
		double * loaded_data = val_arr.data<double>();

		assert(val_arr.word_size == sizeof(double));
		assert(val_arr.shape.size() == 3);


		

		
		//std::vector<std::vector<double>>* values = val_arr.data<std::vector<std::vector<double>>>();
		values.resize(length);


		for(size_t row = 0; row < length; row++) {
			values[row].resize(length);
			for(size_t col = 0; col < length; col++) {
				values[row][col].resize(length);
				for(size_t depth = 0; depth < length; depth++) {
					//std::cout << row*length*length + col*length + depth << std::endl;
					//std::cout << "Vector size: " << values.size() << ", subvector size:" << values[row].size() << std::endl; 
					//std::cout << "Undervector size: " << values[row][col].size() << std::endl;
					values[row][col][depth] = loaded_data[row*length*length + col*length + depth];
				}
			}
		}



		
		//Setup range

		//std::string fname2 = "/home/coraor/TICG-chromatin/2drange.npy";
		std::string fname2 = "/project2/depablo/coraor/fiber_builder/ranges_" +
			std::to_string(nrl) + "_kde.dat.npy";
		cnpy::NpyArray ran_arr = cnpy::npy_load(fname2);

		assert(ran_arr.word_size == sizeof(double));
		assert(ran_arr.shape.size() == 2);

		//read data out from range
		loaded_data = ran_arr.data<double>();

		//Set up O(1) indexing
		size_t ran_num = ran_arr.shape[1];
		//std::cout << "Range size: " << ran_num << std::endl;
		alpha_int = (loaded_data[ran_num-1] - loaded_data[0])/(length-1);
		beta_int = (loaded_data[2*ran_num-1] - loaded_data[ran_num])/(length-1);
		alphaprime_int = (loaded_data[3*ran_num-1] - loaded_data[2*ran_num])/(length-1);
		
		alpha_min = loaded_data[0];
		alpha_max = loaded_data[ran_num-1];
		beta_min = loaded_data[ran_num];
		beta_max = loaded_data[2*ran_num-1];
		alphaprime_min = loaded_data[2*ran_num];
		alphaprime_max = loaded_data[3*ran_num-1];

		//std::cout << "Various parameters: " << alpha_int << beta_int << alphaprime_int;
		//std::cout << std::endl << alpha_min << alpha_max << beta_min << beta_max;
		//std::cout << std::endl << alphaprime_min << alphaprime_max << std::endl;
	}

	//Calculate the interpolated triplet energy 
	double energy(double alpha, double beta, double alphaprime)
	{
		//Calculate grid interpolation
		//std::cout << "Calculating energy" << std::endl;
		//Every point is surrounded by 8 corners. Do a weighted-average of the corners.
		double alpha_point,beta_point,alphaprime_point;
		double final_val = 0.0;

		alpha_point = (alpha-alpha_min)/alpha_int;
		beta_point = (beta-beta_min)/beta_int;
		alphaprime_point = (alphaprime-alphaprime_min)/alphaprime_int;


		if(alpha_point < 0.0 || beta_point < 0.0 || alphaprime_point < 0.0){
			//std::cout << "alpha, beta, alpha': " << alpha << ", " << beta
			//<< ", " << alphaprime << std::endl;
			double dist = (alpha-100)*(alpha-100) + (beta-80.0)*(beta-80.0) + (
				alphaprime-100)*(alphaprime-100);
			return std::cbrt(std::numeric_limits<double>::max())* dist;
		} else if(alpha > alpha_max || beta > beta_max || alphaprime > alphaprime_max){
			//std::cout << "alpha, beta, alpha': " << alpha << ", " << beta
			//<< ", " << alphaprime << std::endl;
			double dist = (alpha-100)*(alpha-100) + (beta-80.0)*(beta-80.0) + (
				alphaprime-100)*(alphaprime-100);
			return std::cbrt(std::numeric_limits<double>::max()) * dist;
		}

		//std::cout << "Indices: " << alpha_point << ", " << beta_point << ", " << alphaprime_point << std::endl;
		std::vector<double> pointvals;
		std::vector<double> weights;
		pointvals.resize(8);
		weights.resize(8);

		double alpha_prop = fmod(alpha_point,1.0);
		double beta_prop = fmod(beta_point,1.0);
		double alphaprime_prop = fmod(alphaprime_point,1.0);
		//std::cout << alpha_prop << ", " << beta_prop << ", " << alphaprime_prop << std::endl;
		

		for(int i=0; i<2; i++){
			for(int j=0; j<2; j++){
				for(int k=0; k<2; k++){
					weights[i*4+j*2+k] = (1-i-(1-2*i)*alpha_prop)*(1-j-(1-2*j)*
						beta_prop)*(1-k-(1-2*k)*alphaprime_prop);
					int x = (int) (alpha_point + i);
					int y = (int) (beta_point + j);
					int z = (int) (alphaprime_point + k);
					//std::cout << x <<','<<y<<','<<z<<std::endl;
					if(x >= values.size() || y >= values.size() || z >= values.size() ||
						x < 0 || y < 0 || z < 0 || i*4+j*2+k > pointvals.size()-1 || i*4+j*2+k < 0){
						std::cout << "Invalid values lookup: " << x << ",";
						std::cout << y << "," << z << "; Returning inf" << std::endl;
						std::cout << "NRL: " << NRL << ", Lengths: " << alpha << ",";
						std::cout << beta << "," << alphaprime << std::endl;
					}
					pointvals[i*4+j*2+k] = values[x][y][z];
				}
			}
		}

		for(size_t i=0;i<8;i++){
			final_val += weights[i]*pointvals[i];
		}

		return final_val;
	}
};


//Lookuptable for energy, first condition
class LookupTable2D {
public:

	int NRL;
	std::vector<std::vector<double>> values;

	double alpha_min, alpha_max, alphaprime_min,alphaprime_max;
	double alpha_int, alphaprime_int;

	cnpy::NpyArray val_arr;
	
	size_t length = 200;

	LookupTable2D() {
		//std::cout << "Warning! Empty 2D Lookuptable generated!" << std::endl;
	}



	LookupTable2D(int const & nrl)
		: NRL{nrl} {


		std::string fname = "/project2/depablo/coraor/fiber_builder/first_cond_" + 
				std::to_string(nrl) + "_kde.dat.npy";
		cnpy::NpyArray val_arr = cnpy::npy_load(fname);
		double * loaded_data = val_arr.data<double>();

		assert(val_arr.word_size == sizeof(double));
		assert(val_arr.shape.size() == 2);
		
		values.resize(length);

		for(size_t row = 0; row < length; row++) {
			values[row].resize(length);
			for(size_t col = 0; col < length; col++) {
				values[row][col] = loaded_data[row*length + col];
			}
		}



		
		//Setup range

		std::string fname2 = "/project2/depablo/coraor/fiber_builder/ranges_" +
			std::to_string(nrl) + "_kde.dat.npy";
		cnpy::NpyArray ran_arr = cnpy::npy_load(fname2);

		assert(ran_arr.word_size == sizeof(double));
		assert(ran_arr.shape.size() == 2);

		//read data out from range
		loaded_data = ran_arr.data<double>();

		//Set up O(1) indexing
		size_t ran_num = ran_arr.shape[1];
		//std::cout << "Range size: " << ran_num << std::endl;
		alpha_int = (loaded_data[ran_num-1] - loaded_data[0])/(length-1);
		alphaprime_int = (loaded_data[3*ran_num-1] - loaded_data[2*ran_num])/(length-1);
		
		alpha_min = loaded_data[0];
		alpha_max = loaded_data[ran_num-1];
		alphaprime_min = loaded_data[2*ran_num];
		alphaprime_max = loaded_data[3*ran_num-1];

		//std::cout << "Various parameters: " << alpha_int << beta_int << alphaprime_int;
		//std::cout << std::endl << alpha_min << alpha_max << beta_min << beta_max;
		//std::cout << std::endl << alphaprime_min << alphaprime_max << std::endl;
		
	
	}

	//Calculate the interpolated triplet energy 
	double energy(double alpha, double alphaprime)
	{
		//Calculate grid interpolation
		//std::cout << "Calculating energy" << std::endl;
		//Every point is surrounded by 8 corners. Do a weighted-average of the corners.
		
		//Convert to Angstroms

		double alpha_point, alphaprime_point;
		double final_val = 0.0;

		alpha_point = (alpha-alpha_min)/alpha_int;
		alphaprime_point = (alphaprime-alphaprime_min)/alphaprime_int;

		if(alpha_point < 0.0 || alphaprime_point < 0.0){
			double dist = (alpha-100)*(alpha-100)+ (alphaprime-100)*(alphaprime-100);
			return sqrt(std::numeric_limits<double>::max()) * dist;
		} else if(alpha > alpha_max || alphaprime_point > alphaprime_max){
			double dist = (alpha-100)*(alpha-100) + (alphaprime-100)*(alphaprime-100);
			return sqrt(std::numeric_limits<double>::max()) * dist;
		}

		//std::cout << "Indices: " << alpha_point << ", " << beta_point << ", " << alphaprime_point << std::endl;
		std::vector<double> pointvals;
		std::vector<double> weights;
		pointvals.resize(4);
		weights.resize(4);

		double alpha_prop = fmod(alpha_point,1.0);
		double alphaprime_prop = fmod(alphaprime_point,1.0);
		//std::cout << alpha_prop << ", " << beta_prop << ", " << alphaprime_prop << std::endl;

		for(int i=0; i<2; i++){
			for(int k=0; k<2; k++){
				weights[i*2+k] = (1-i-(1-2*i)*alpha_prop)*(1-k-(1-2*k)*alphaprime_prop);
				int x = (int) (alpha_point + i);
				int z = (int) (alphaprime_point + k);
				//std::cout << x <<','<<y<<','<<z<<std::endl;
				pointvals[i*2+k] = values[x][z];
			}
			
		}

		for(size_t i=0;i<4;i++){
			final_val += weights[i]*pointvals[i];
		}

		return final_val;
	}
};

//Lookuptable for dihedrals.
class LookupTable1D {
public:

	int NRL;
	std::vector<double> values;

	double dih_min, dih_max;
	double dih_int;

	cnpy::NpyArray val_arr;
	
	size_t length = 200;

	LookupTable1D() {
		//std::cout << "Warning! Empty 1D Lookuptable generated!" << std::endl;
	}

	LookupTable1D(int const & nrl)
		: NRL{nrl} {

		//std::cout << "making 1d lookup table" << std::endl;

		std::string fname = "/project2/depablo/coraor/fiber_builder/dih_energies_" +
			std::to_string(nrl) + ".npy";
		cnpy::NpyArray val_arr = cnpy::npy_load(fname);
		double * loaded_data = val_arr.data<double>();

		assert(val_arr.word_size == sizeof(double));
		assert(val_arr.shape.size() == 1);
		
		values.resize(length);

		for(size_t row = 0; row < length; row++) {
			values[row] = loaded_data[row];
		}
		
		//Setup range

		std::string fname2 = "/project2/depablo/coraor/fiber_builder/dih_ranges_" +
			std::to_string(nrl) + ".npy";
		cnpy::NpyArray ran_arr = cnpy::npy_load(fname2);

		assert(ran_arr.word_size == sizeof(double));
		assert(ran_arr.shape.size() == 1);

		//read data out from range
		loaded_data = ran_arr.data<double>();

		//Set up O(1) indexing
		size_t ran_num = ran_arr.shape[0];
		//std::cout << "Range size: " << ran_num << std::endl;
		dih_int = (loaded_data[ran_num-1] - loaded_data[0])/(length-1);
		
		
		dih_min = loaded_data[0];
		dih_max = loaded_data[ran_num-1];
		//std::cout << "Various parameters: " << alpha_int << beta_int << alphaprime_int;
		//std::cout << std::endl << alpha_min << alpha_max << beta_min << beta_max;
		//std::cout << std::endl << alphaprime_min << alphaprime_max << std::endl;
	}

	//Calculate the dihedral energy. dih in radians.
	double energy(double const &dih)
	{
		//Calculate grid interpolation
		//std::cout << "Calculating energy" << std::endl;
		//Every point is surrounded by 8 corners. Do a weighted-average of the corners.
		double dih_point;
		double final_val = 0.0;

		dih_point = (dih-dih_min)/dih_int;

		if(dih_point < 0.0 || dih > dih_max){
			return sqrt(std::numeric_limits<double>::max());
		}

		//std::cout << "Indices: " << alpha_point << ", " << beta_point << ", " << alphaprime_point << std::endl;
		std::vector<double> pointvals;
		std::vector<double> weights;
		pointvals.resize(2);
		weights.resize(2);

		double dih_prop = fmod(dih_point,1.0);
		//std::cout << alpha_prop << ", " << beta_prop << ", " << alphaprime_prop << std::endl;

		for(int i=0; i<2; i++){
				weights[i] = (1-i-(1-2*i)*dih_prop);
				int x = (int) (dih_point + i);
				//std::cout << x <<','<<y<<','<<z<<std::endl;
				pointvals[i] = values[x];
			}
			
		

		for(size_t i=0;i<2;i++){
			final_val += weights[i]*pointvals[i];
		}

		return final_val;
	}
};




//Trinucleosome energy. 
class Fiber_Angle : public Angle {
public:
	//dyad should be a pointer to nrl2's alpha, alphaprime energy.
	Fiber_Angle(Bead * bead1, Bead * bead2, Bead * bead3, LookupTable3D * LT1, 
		LookupTable3D * LT2, LookupTable2D * LT3)
		: Angle{bead1,bead2,bead3}, triad1{LT1}, triad2{LT2}, dyad{LT3} {}

	LookupTable3D* triad1;
	LookupTable3D* triad2;
	LookupTable2D* dyad;
	//lh = True if bead1 has a LH bound
	



	double prior_energy;

	double energy()
	{
		Eigen::RowVector3d disp1 = pbead1->r - pbead2->r;
		Eigen::RowVector3d disp2 = pbead3->r - pbead2->r;
		Eigen::RowVector3d disp3 = pbead3->r - pbead1->r;

		double alpha = sqrt(disp1.dot(disp1));
		double alphaprime = sqrt(disp2.dot(disp2));
		double beta = sqrt(disp3.dot(disp3));

		double triad_en = 0.0;
		//Convert nm --> Angstroms
		alpha *= 10;
		beta *= 10;
		alphaprime *= 10;
		if(triad1 == triad2){
			triad_en = triad1->energy(alpha,beta,alphaprime);
		} else{
			triad_en = (triad1->energy(alpha,beta,alphaprime) + triad2->energy(alpha,beta,alphaprime))/2.0;	
		}
		double dyad_en = dyad->energy(alpha,alphaprime);
		//std::cout << "Triad_energy: " << triad_en << ", dyad energy: " << dyad_en << std::endl;
		//if(triad_en + dyad_en > prior_energy + 100){
		//	std::cout << "dramatic triplet energy increase at " << pbead1->id << "," << pbead2->id
		//	<< "," << pbead3->id << std::endl;
		//}
		prior_energy = triad_en + dyad_en;

		//Check beads for LH, apply potentials based on this.

		int total_lhs = 0;
		total_lhs += pbead1->lh_bound;
		total_lhs += pbead2->lh_bound;
		total_lhs += pbead3->lh_bound;
		if(total_lhs > 0){
			
			double lhslope = 0.2*total_lhs/3.0;
			if (beta > 100){
				prior_energy += (beta-100.0)*lhslope;
			}

			if(pbead2->lh_bound){

				// second NRL leg
				if (alphaprime > 100){
					//add linear A(r) based on average slope increase: 1.222 kT/nm above
					//10 nm
					//Slope is parabolic function of NRL, pulled from dinuc data:
					// U = (a-100)*(aNRL^2+bNRL+c)
					lhslope = -0.00010907*(triad2->NRL)*(triad2->NRL) + 0.04089813*(triad2->NRL)+(
						-3.705485);
					prior_energy += (alphaprime-100.0)*lhslope;
				} 
				// first NRL leg
				if (alpha > 100){
					lhslope = -0.00010907*(triad1->NRL)*(triad1->NRL) + 0.04089813*(triad1->NRL)+(
						-3.705485);
				prior_energy += (alpha-100.0)*lhslope;} 			
			}
		}

		return prior_energy;
	}

};

//Tetranucleosome dihedral energy. 
class Fiber_Dihedral : public Dihedral {
public:
	//dyad should be a pointer to nrl2's alpha, alphaprime energy.
	Fiber_Dihedral(Bead* bead1, Bead* bead2, Bead* bead3, Bead* bead4, LookupTable1D* LT1, 
		LookupTable1D* LT2, LookupTable1D* LT3)
		: Dihedral{bead1,bead2,bead3,bead4}, tetrad1{LT1}, tetrad2{LT2}, tetrad3{LT3} {}

	LookupTable1D* tetrad1;
	LookupTable1D* tetrad2;
	LookupTable1D* tetrad3;

	double energy()
	{
		Eigen::RowVector3d b1 = pbead2->r - pbead1->r;
		Eigen::RowVector3d b2 = pbead3->r - pbead2->r;
		Eigen::RowVector3d b3 = pbead4->r - pbead3->r;

		Eigen::RowVector3d n1 = b1.cross(b2);
		Eigen::RowVector3d n2 = b2.cross(b3);
		n1 /= sqrt(n1.dot(n1));

		Eigen::RowVector3d m1 = n1.cross(b2/sqrt(b2.dot(b2)));

		double x = n1.dot(n2);
		double y = m1.dot(n2);
		double dih = atan2(y,x)*180.0/M_PI;


		double tetrad_en = (tetrad1->energy(dih) + tetrad2->energy(dih) + tetrad3->energy(dih))/3.0;
		return tetrad_en;
	}

};


class Cell {
public:
	Eigen::RowVector3d r; // corner of cell == CURRENTLY UNUSED
	std::unordered_set<Bead*> contains; // beads associated inside this gridpoint
	double phi;
	double vol;
	int local_HP1; // local number of HP1
	static const int beadvol = 520; // volume of a bead in the cell. 
	static constexpr double vint = 4/3*M_PI*3*3*3; // volume of HP1 interaction
	static constexpr double J = -4;  // kT 

	//Ideal Chromatin Constants
	static constexpr double gamma1 = -0.030;
	static constexpr double gamma2 = -0.351;
	static constexpr double gamma3 = -3.727;
	static constexpr double d_scale = 200./50000; //Rescale d in TICG to Michrom

	static int ntypes;
	std::vector<int> typenums = std::vector<int>(ntypes); // always up-to-date
	std::vector<double> phis = std::vector<double>(ntypes); // only up-to-date after energy calculation

	void print() 
	{
		std::cout << r << "     N: " << contains.size() << std::endl;
		for (Bead* bead : contains)
		{
			//std::cout << "With beads: " << contains.size() << std::endl;
			//bead->print();
		};

	}

	void reset()
	{
		// clears population trackers
		contains.clear();
		std::fill(typenums.begin(), typenums.end(), 0);  // DO NOT USE .clear()
		std::fill(phis.begin(), phis.end(), 0);      // ... it doesn't re-assign to 0's
	}

	void moveIn(Bead* bead)
	{
		// updates local number of each type of bead, but does not recalculate phis
		contains.insert(bead);
		local_HP1 += bead->nbound();
		//typenums += bead->d;

		for(int i=0; i<ntypes; i++)
		{
			typenums[i] += bead->d[i];
		}
	}
		
	void moveOut(Bead* bead)
	{
		// updates local number of each type of bead, but does not recalculate phis
		contains.erase(bead);
		local_HP1 -= bead->nbound();
		//typenums -= bead->d;

		for(int i=0; i<ntypes; i++)
		{
			typenums[i] -= bead->d[i];
		}
	}

	double getEnergy(const Eigen::MatrixXd &chis,bool ideal_chrom=false)
	{
		float phi_solvent = 1 - contains.size()*beadvol/vol;

		for (int i=0; i<ntypes; i++)
		{
			phis[i] = typenums[i]*beadvol/vol;
			//phi_solvent -= phis[i]; // wrong!! when nucl. have multiple marks
		}

		double U = 0;

		if (phi_solvent < 0.5)
		{
			// high volume fraction occurs when more than 50% of the volume is occupied by beads
			U = 99999999999;
		}
		else 
		{
			for (int i=0; i<ntypes; i++)
			{
				for (int j=i; j<ntypes; j++)
				{
					if (i==j)
					{
						// A - Solvent
						//U += chis(i,j)*phis[i]*phi_solvent*vol/beadvol;

						// A - self
						//U += -1*chis(i,j)*phis[i]*phis[j]*vol/beadvol;
						
						// the old way
						U += chis(i,j)*phis[i]*phis[j]*vol/beadvol;

					}
					else
					{
						U += chis(i,j)*phis[i]*phis[j]*vol/beadvol;
						//std::cout << chis(i,j) << " " << phis[i] << " " << phis[j] << " " << vol << " " << beadvol << std::endl;
					}
				}
			}
			if(ideal_chrom){
				//Calculate pairwise ideal chromatin term
				double d; //difference in index
				int imax = (int) contains.size();
				std::vector<int> inds;
				for (const auto& elem: contains) {
					inds.push_back(elem->id);
				}

				for (int i=0; i < imax-1; i++){
					for(int j=i+1; j < imax; j++){
						d = abs(inds[i] - inds[j]);
						//Ideal chromatin:
						d *= d_scale;
						if (d < 500 && d > 3){
							U += gamma1/log(d) + gamma2/d + gamma3/(d*d);
						}
					}

				}
				/*
				for (int i=0; i < imax-1; i++){
					for (int j = i+1; j < imax; j++){
						d = abs(contains[i]->id - contains[j]->id);
						//Ideal chromatin:
						d *= d_scale;
						if (d < 500 && d > 3){
							U += gamma1/log(d) + gamma2/d + gamma3/(d*d);
						}
					}
				}
				*/
			}
		}

		return U;
	}

	double getTailEnergy()
	{
		double rho = local_HP1/vol;
		double U = J/2*vint*vol*rho*rho;
		return  U;
	}

};


class Grid {
public:
	std::vector<std::vector<std::vector<Cell>>> cells; 
	std::unordered_set<Cell*> active_cells;       // cells marked as active (within simulation region)

	double delta;              // grid cell size 

	bool cubic_boundary = true;
	bool spherical_boundary = false;

	int L;                     // size of cubic boundary in units of grid cells
	double radius;              // radius of simulation volume in [nanometers]
	int boundary_radius;          // radius of boundary in units of grid cells
	Eigen::RowVector3d sphere_center; // center of spherical boundary

	// bottom-left-most grid cell: 
	// With grid moves on, it will diffuse with periodic boundaries
	// inside the volume bounded by (-delta, -delta, -delta) and (0,0,0) 
	Eigen::RowVector3d origin; 

	void generate()
	{
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
					
					cells[i][j][k].phi = 0.0;
					cells[i][j][k].vol = delta*delta*delta;
					cells[i][j][k].r = {i*delta,j*delta,k*delta};
					//cells[i][j][k].print();
				}
			}
		}
	}

	void setActiveCells()
	{
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
	}
 	
	void printActiveCells()
	{
		// prints only the active cells that contains particles
		std::cout << "Printing Active Cells: number:" << active_cells.size() << std::endl;
		for(Cell* cell : active_cells) {
			if(cell->contains.size() > 0) cell->print();
		}
	}


	void meshBeads(std::vector<Bead> &beads)
	{
		// Inserts all beads into their corresponding grid cells
		for(int i=0; i<cells.size(); i++) {
			for(int j=0; j<cells[i].size(); j++) {
				for(int k=0; k<cells[i][j].size(); k++) {
					cells[i][j][k].reset();
				}
			}
		}

		int i, j, k;
		for(Bead& bead : beads)   // the reference is crucial-- otherwise, copies are made in the for-each loop
		{
			i = floor((bead.r(0) - origin(0))/delta);
			j = floor((bead.r(1) - origin(1))/delta);
			k = floor((bead.r(2) - origin(2))/delta);

			cells[i][j][k].moveIn(&bead);
		}
	}

	Cell* getCell(const Bead& bead)
	{
		// Returns a pointer to the cell in which a bead is located
		int i, j, k;
		i = floor((bead.r(0) - origin(0))/delta);
		j = floor((bead.r(1) - origin(1))/delta);
		k = floor((bead.r(2) - origin(2))/delta);
		return &cells[i][j][k];
	}

	Cell* getCell(const Eigen::RowVector3d &r)
	{
		// Returns a pointer to the cell in which a bead is located
		int i, j, k;
		i = floor((r(0) - origin(0))/delta);
		j = floor((r(1) - origin(1))/delta);
		k = floor((r(2) - origin(2))/delta);
		return &cells[i][j][k];
	}

	bool checkCellConsistency(int nbeads)
	{
		// checks to see if number of beads in the grid is the same as simulation nbeads
		double cellbeads = 0;
		for(Cell* cell : active_cells)
		{
			cellbeads += cell->contains.size();
		}

		return (cellbeads == nbeads); 
	}

	bool checkHp1Consistency(int hp1_tot);
			
	double energy(const std::unordered_set<Cell*>& flagged_cells, const Eigen::MatrixXd &chis, bool ideal_chrom)
	{
		// nonbonded volume interactions
		double U = 0; 
		for(Cell* cell : flagged_cells)
		{
			U += cell->getEnergy(chis,ideal_chrom);
		}
		return U;
	}

	double tailEnergy(const std::unordered_set<Cell*>& flagged_cells)
	{
		// HP1 oligomerization volume interactions
		double U = 0;
		for(Cell* cell : flagged_cells)
		{
			U += cell->getTailEnergy();
		}
		return U;
	}

	double get_ij_Contacts(int i, int j) 
	{
		// calculates average phi_i phi_j
		double n = 0;
		for(Cell* cell : active_cells)
		{
			n += cell->phis[i] * cell->phis[j];
		}

		n /= active_cells.size(); 
		return n;
	}
};


class Sim {
public: 
	std::vector<Bead> beads;
	std::vector<Bond*> bonds; // pointers because Bond class is virtual
	std::vector<Angle*> angles;
	std::vector<Dihedral*> dihedrals;

	std::vector<int> nrls;
	int min_nrl = 158;
	std::vector<LookupTable3D> angle_lookups;
	std::vector<LookupTable2D> bond_lookups;
	std::vector<LookupTable1D> dihedral_lookups;
	Grid grid;

	RanMars* rng; 

	double chi; 
	Eigen::MatrixXd chis;
	int nspecies; // number of different epigenetic marks
	int nbeads; 
	double hp1_mean_conc; 
	double hp1_total; // conserved quantity 
	double total_volume;

	double lh_mean_conc;
	double lh_total; // conserved

	static int hp1_free;
	static int lh_free;


	FILE *xyz_out; 
	FILE *energy_out;
	FILE *obs_out;

	// MC variables
	int decay_length; 
	int exp_decay;// = nbeads/decay_length;             // size of exponential falloff for MCmove second bead choice
	int exp_decay_crank;// = nbeads/decay_length;
	int exp_decay_pivot;// = nbeads/decay_length;
	
	//New
	double step_disp = 10;
	double step_trans = 10;

	//Originals
	//double step_disp = 1;
	//double step_trans = 2;

	double step_crank = M_PI/6;
	double step_pivot = M_PI/6;
	double step_rot = M_PI/12;
	double step_grid; // based off fraction of delta, see initialize

	int n_disp;// = 0;
	int n_trans;// = decay_length; 
	int n_crank;// = decay_length;
	int n_pivot;// = decay_length;
	int n_bind;// = 0;
	int n_rot;// = nbeads;
	int n_link;// = nbeads;

	int acc_trans = 0;
	int acc_crank = 0;
	int acc_pivot = 0;
	int acc_rot = 0;
	int acc_bind = 0;
	int acc_disp = 0;
	int acc_link = 0;

	bool production; // if false, equilibration
	int nSteps;// = n_trans + n_crank + n_pivot + n_rot + n_bind;
	int nSweeps;
	int dump_frequency; // dump every x sweeps
	int dump_stats_frequency;
	//int nEquilibSweeps; // = 10*dump_frequency;
	int acc = 0;

	bool bonded_on; 
	bool nonbonded_on;
	bool binding_on; 
	bool lh_on;
	bool ideal_chrom;

	bool gridmove_on;

	bool AB_block; // = true;
	int domainsize = nbeads;

	bool load_chipseq;
	bool load_configuration;
	std::string load_configuration_filename; 
	std::string load_chipseq_filename; 
	std::vector<std::string> chipseq_files;

	std::string chipseq_1;
	std::string chipseq_2;
	std::string chipseq_3;
	std::string chipseq_4;
	std::string chipseq_5;
	std::string chipseq_6;
	std::string chipseq_7;
	
	// analytics
	bool print_MC; // = false;
	bool print_trans; // = false;
	bool print_acceptance_rates; // = true;
	bool timers; // = false;

	std::vector<std::vector<int>> contact_map;
	int contact_resolution; //= 500;

	void setupContacts()
	{
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

	//Recount the number of HP1s in each cell based on number of actual bound
	//tails
	void correct_hp1()
	{
		for(Cell* cell : grid.active_cells)
		{
			int hp1_tot = 0;
			for(Bead* bead1 : cell->contains)
			{
				hp1_tot += bead1->nbound();
			}
			cell->local_HP1 = hp1_tot;
		}
	}

	void updateContacts()
	{
		for(Cell* cell : grid.active_cells)
		{
			for(Bead* bead1 : cell->contains)
			{
				for(Bead* bead2 : cell->contains)
				{
					int id1 = bead1->id/contact_resolution;
					int id2 = bead2->id/contact_resolution;
					contact_map[id1][id2] += 1;
				}
			}
		}
	}

	double getLHChemPot()
	{
		//LH volume approximated by taking 1.35 g/mL as the density of 
		//"protein", and using this to calculate the solvent-excluded-volume
		//of the protein.
		//20731.740000 is the mass from the internet, lammps units
		//15,356.84 mL/mole
		//2.550122e-20 mL/particle
		//Make reference conc 1 mM,
		return log(lh_free/total_volume/ (602));
	}

	double getChemPot()
	{
		// chemical potential based on [HP1_free] in micromolar. 
		// Conversion ratio: (HP1_free/um^3) / [uM] = 602
		return log(hp1_free/total_volume / (602));
	}

	Eigen::MatrixXd unit_vec(Eigen::MatrixXd b)
	{
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

	void readInput()
	{
		std::cout << "reading input file ... ";

		std::ifstream i("config.json");
		nlohmann::json config;
		i >> config;
		
		nspecies = config["nspecies"];
		load_chipseq = config["load_chipseq"];

		if (load_chipseq)
		{
			chis = Eigen::MatrixXd::Zero(nspecies, nspecies); 

			char first = 'A' + 1;
			for (int i=0; i<nspecies; i++)
			{

				// should be included even if load_chipseq is false... fix later

				for (int j=i; j<nspecies; j++)
				{
					//std::cout << i << ", " << j << std::endl;
					char first = 'A' + i;
					char second = 'A' + j;
					//std::cout << first << ", " << second << std::endl;
					std::string chistring = "chi";
					chistring += first;
					chistring += second;
					std::cout << chistring << std::endl;
					chis(i,j) = config[chistring];
					std::cout << chistring << " " << chis(i,j) << std::endl;
				}
			}

			for (auto file : config["chipseq_files"])
			{
				chipseq_files.push_back(file);
			}
			std::cout << chipseq_files.size();

			nspecies = chipseq_files.size();
			Cell::ntypes = nspecies;
			Bead::ntypes = nspecies;
		}

		gridmove_on = config["gridmove_on"];
	
		std::cout << "made it out" << std::endl;
		production = config["production"];
		nbeads = config["nbeads"];
		hp1_mean_conc = config["hp1_mean_conc"];
		lh_mean_conc = config["lh_mean_conc"];
		decay_length = config["decay_length"];
		nSweeps = config["nSweeps"];
		dump_frequency = config["dump_frequency"];
		dump_stats_frequency = config["dump_stats_frequency"];
		bonded_on = config["bonded_on"];
		nonbonded_on = config["nonbonded_on"];
		binding_on = config["binding_on"];
		lh_on = config["lh_on"];
		AB_block = config["AB_block"];
		domainsize = config["domainsize"];
		timers = config["timers"];
		load_configuration = config["load_configuration"];
		load_configuration_filename = config["load_configuration_filename"];
		load_chipseq_filename = config["load_chipseq_filename"];
		print_MC = config["print_MC"];
		print_trans = config["print_trans"];
		print_acceptance_rates = config["print_acceptance_rates"];
		contact_resolution = config["contact_resolution"];
		ideal_chrom = config["ideal_chrom"];

		int seed = config["seed"];
		rng = new RanMars(seed);

		std::cout << "read successfully" << std::endl;
	}

	void makeOutputFiles()
	{
		std::cout << "making output files ... ";

		// make and populate data output directory
		const int dir_err = system("mkdir data_out");
		if (dir_err == -1) {std::cout << "error making data_out directory" << std::endl;}
	        xyz_out = fopen("./data_out/output.xyz", "w");
		energy_out = fopen("./data_out/energy.traj", "w");
		obs_out = fopen("./data_out/observables.traj", "w");

		std::cout << "created successfully" << std::endl;
	}

	bool outside_boundary(Eigen::RowVector3d r)
	{
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

		
	void initialize()
	{
		std::cout << "Initializing simulation objects ... " << std::endl;
		Timer t_init("Initializing");

		grid.delta = 28.7; // grid cell size nm
		step_grid = grid.delta/10.0; // size of grid displacement MC moves

		double Vbar = 7765.77;  // nm^3/bead: reduced number volume per spakowitz: V/N
		double vol = Vbar*nbeads; // simulation volume in nm^3
		grid.L= std::round(std::pow(vol,1.0/3.0) / grid.delta); // number of grid cells per side // ROUNDED, won't exactly equal a desired volume frac
		std::cout << "grid.L is: " << grid.L << std::endl;
		total_volume = pow(grid.L*grid.delta/1000.0, 3); // micrometers^3 ONLY TRUE FOR CUBIC SIMULATIONS 

		grid.radius = std::pow(3*vol/(4*M_PI), 1.0/3.0); // radius of simulation volume
		grid.boundary_radius = std::round(grid.radius); // radius in units of grid cells
		// sphere center needs to be centered on a multiple of grid delta
		grid.sphere_center = {grid.boundary_radius*grid.delta, grid.boundary_radius*grid.delta, grid.boundary_radius*grid.delta};

		hp1_free = hp1_mean_conc*total_volume* 602; // 602 = number/(uM*um^3)
		hp1_total = hp1_free;


		//Internal calibration: want lh ~ 1/10th of nbeads
		//double ideal_lh = std::pow(1000,3)/(4*602)
		lh_free = lh_mean_conc*total_volume* 602;
		lh_total = lh_free; 
		std::cout << "Linker histone molecules: " << lh_free << std::endl;

		exp_decay = nbeads/decay_length;             // size of exponential falloff for MCmove second bead choice
		exp_decay_crank = nbeads/decay_length;
		exp_decay_pivot = nbeads/decay_length;

		
		n_trans = decay_length; 
		//Massively increased n_trans:
		//n_trans = 50*decay_length;
		n_crank = decay_length;
		n_pivot = decay_length/10;

		n_disp = nbeads;
		n_bind = nbeads;
		n_link = nbeads;
		n_rot = 0;
		//n_rot = nbeads;
		nSteps = n_trans + n_crank + n_pivot + n_rot + n_bind + n_disp + n_link;

		//Initialize lookuptables
		angle_lookups.resize(207-min_nrl+1);
		bond_lookups.resize(207-min_nrl+1);
		dihedral_lookups.resize(207-min_nrl+1);
		std::cout << "Creating lookup tables..." << std::endl;
		for(int i=min_nrl;i<208;i++){
			std::cout << "Made lookuptable for nrl " << i << std::endl;
			angle_lookups[i-min_nrl] = LookupTable3D(i);
			bond_lookups[i-min_nrl] = LookupTable2D(i);
			dihedral_lookups[i-min_nrl] = LookupTable1D(i);
		}
		std::cout << "Lookup tables completed." << std::endl;


		double bondlength = 16.5;
		beads.resize(nbeads);  // uses default constructor initialization to create nbeads;

		std::cout << "load configuratiOn is " << load_configuration << std::endl;
		// set configuration
		if(load_configuration) {

			std::cout << "Loading configuration from " << load_configuration_filename << std::endl;
			std::ifstream IFILE; 
			IFILE.open(load_configuration_filename); 
			std::string line;
			getline(IFILE, line); // nbeads line
			std::cout << line << std::endl;
			getline(IFILE, line); // comment line 
			std::cout << line << std::endl;
				
			// first bead
			getline(IFILE, line);
			std::cout << line << std::endl;
			std::stringstream ss;
			ss << line;
			ss >> beads[0].id;
			ss >> beads[0].r(0);
			ss >> beads[0].r(1);
			ss >> beads[0].r(2);

			for(int i=1; i<nbeads; i++)
			{
				getline(IFILE, line);
				std::cout << line << std::endl;
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
		else {
			// RANDOM COIL 
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

		// set epigenetic sequence
		if (load_chipseq) {
			int marktype = 0;
			std::cout << "Loading chipseq" << std::endl;
			for (std::string chipseq_file : chipseq_files)
			{
				std::ifstream IFCHIPSEQ;
				int tail_marked;
				IFCHIPSEQ.open(chipseq_file);
				
				//mark the bead with its type as well as its 
				for(int i=0; i<nbeads; i++)
				{
					IFCHIPSEQ >> tail_marked;
					if (tail_marked == 1)
					{
						beads[i].d[marktype] = 1;
					}
					else if (tail_marked == 2){
						beads[i].d[marktype] = 2;
					}
					else
					{
						beads[i].d[marktype] = 0;
					}
				}
				marktype++;
				IFCHIPSEQ.close();
				
				// assigns methylation marks based on chipseq data 
				int ntails_methylated;
				for(int i=0; i<nbeads; i++) {
					IFCHIPSEQ >> ntails_methylated;
					std::cout << "Methylating tails for bead " <<i << std::endl;
					if (ntails_methylated == 2) {
						beads[i].tail1.mark = 1;
						beads[i].tail2.mark = 1;
						beads[i].tails_methylated = 2;
					}
					else if (ntails_methylated == 1) {
						beads[i].tail1.mark = 1;
						beads[i].tails_methylated = 1;
					}
				}
				
			}
		}
		else {
			// assigns methylation marks in a block polymer manner
			int blocksize  = nbeads/5;
			for(int i=0; i<nbeads; i++) {
				if (i%blocksize < blocksize/3) {
					beads[i].tail1.mark = 1;
					beads[i].tail2.mark = 1;
					beads[i].tails_methylated = 2;
				}
				else if (i%blocksize < 2*blocksize/3) {
					beads[i].tail1.mark = 1;
					beads[i].tails_methylated = 1;
				}
			}
		}

        // set domain sequence
		if (AB_block) {
			for(int i=0; i<nbeads; i++)
			{
				//if (i%domainsize < domainsize/2.0)
				//{
				//	beads[i].d = Bead::A;
				//}
				//else
				//{
				//	beads[i].d = Bead::B;
				//}

				double sig = 500.0;
				double prob = gaussian(2000.0, sig, i) + gaussian(6000.0, sig, i);
				double rndDouble = (double)rand() / RAND_MAX;

				if(rndDouble < prob)
				{
					beads[i].d[0] = 1; //Bead::A;
				}
				else
				{
					beads[i].d[1]= 1; // Bead::B;
				}
			}
		}

		//Set nrl distribution
		nrls.resize(nbeads-1);
		std::vector<double> probs, cum_probs;
		double nrl,prob;
		double ktw = 13.9; // ktwist from "A relationship between the helical twist of DNA and the ordered positioning of nucleosomes in all eukaryotic cells"


		probs.resize(207-min_nrl+1);
		cum_probs.resize(probs.size());
		double psum = 0.0;
		//Calculate the probability distribution & cumulative probability
		for(int i=0; i < probs.size(); i++){
			nrl = i + min_nrl + 0.0;
			//Generate according to Widom 1992 distribution
			probs[i] = 1.0/(20.0*sqrt(2*M_PI))*exp(-0.5*(nrl-200.0)*(nrl-200.0)/(20.0*20.0));
			
			//Multiply by positive + negative twist energies
			probs[i] *= (exp(-ktw*pow(((i+min_nrl-147) % 10)/10.0,2))+
				exp(-ktw*pow(1.-(i+min_nrl-147 % 10)/10.0,2)));
			psum += probs[i];
		}

		probs[0] /= psum;
		cum_probs[0] = probs[0];
		for(int i=1; i < probs.size(); i++){
			probs[i] /= psum;
			cum_probs[i] = cum_probs[i-1] + probs[i];
		}

		assert(cum_probs[cum_probs.size()-1] > 0.99);


		//Draw random numbers and draw from widom distribution
		for(int i=0; i<nbeads-1; i++){
			double seed = rng->uniform();
			//search for first val greater than seed
			int j = 0;
			while(cum_probs[j] < seed){
				j++;
			}
			nrls[i] = min_nrl + j;
		}

		// set bonds
		//For Fiber, only make first bond harmonic
		bonds.push_back(new Harmonic_Bond(&beads[0],&beads[1],1.0,160.0));
		
		/*
		bonds.resize(nbeads-1); // use default constructor
		for(int i=0; i<nbeads-1; i++)
		{
			bonds[i] = new DSS_Bond{&beads[i], &beads[i+1]};
		}
		*/

		//set angles
		angles.resize(nbeads-2);
		for(int i=0; i < nbeads-2; i++){
			angles[i] = new Fiber_Angle(&beads[i], &beads[i+1], &beads[i+2], 
				&angle_lookups[nrls[i]-min_nrl],&angle_lookups[nrls[i+1]-min_nrl],
				&bond_lookups[nrls[i]-min_nrl]);
		}

		//set Dihedrals
		dihedrals.resize(nbeads-3);
		for(int i=0; i < nbeads-3; i++){
			dihedrals[i] = new Fiber_Dihedral(&beads[i],&beads[i+1],&beads[i+2],&beads[i+3],
				&dihedral_lookups[nrls[i]-min_nrl],&dihedral_lookups[nrls[i+1]-min_nrl],
				&dihedral_lookups[nrls[i+2]-min_nrl]);
		}


		// output for vis.
		dumpData();
		//dumpEnergy(0);
		std::cout << "Objects created" << std::endl;
	}

	double gaussian(double mu, double sig, double x)
	{
		// un-normalized gaussian
		return std::exp(-std::pow(x-mu,2)/(2*sig*sig));
	}

	void print()
	{
		std::cout << "simulation in : "; 
		std::cout << "With beads: " << std::endl;
		for(Bead& bb : beads) bb.print();             // use reference to avoid copies
		std::cout << "And bonds: " << std::endl;
		for(Bond* bo : bonds) bo->print();
	}

	double getAllBondedEnergy()
	{
		double U = 0;
		for(Bond* bond : bonds) {U += bond->energy();}
		for(Angle* angle : angles) {U += angle->energy();}
		for(Dihedral* dihedral: dihedrals) {U += dihedral->energy();}
		return U;
	}

	double getBondedEnergy(int first, int last, bool print = false)
	{
		double U = 0;
		//for(Bond* bo : bonds) U += bo->energy();  // inefficient 
		/*
		if(last == first){
			if(last != 0) {U += angles[last-2]->energy()}
		}
		*/
		//Iterate over bonds
		//if(first > 0 && bonds.size() >= first){U += bonds[first-1]->energy();}
		//if(last < nbeads-1 && bonds.size() >= last){U += bonds[last]->energy();}

		//Iterate over angles
		//first
		int min_angle;
		int max_angle;
		int last_min_angle;
		int last_max_angle;
		int min_dih;
		int max_dih;
		int last_min_dih;
		int last_max_dih;

		if(print){std::cout << "first: " << first << ", last: " << last << std::endl;}

		if(first > 0){
			(first-2 > 0) ? (min_angle = first-2) : (min_angle = 0);
			(first-1 < ((int)angles.size())-1) ? (max_angle = first-1) : (max_angle = ((int) angles.size())-1);
			
			for(size_t i=min_angle;i <= max_angle; i++){
				U += angles[i]->energy();
			}
		} else {
			min_angle = -1;
			max_angle = -1;
		}
		if(print){std::cout << "Energy after first: " << U << std::endl;
		std::cout << "min_angle: " << min_angle << std::endl;
		std::cout << "max_angle: " << max_angle << std::endl;
		}
		//last
		if(last < nbeads-1){
			//don't double count angle
			(last - 1 > max_angle+1) ? (last_min_angle = last-1) : (last_min_angle = max_angle+1);
			last_max_angle = std::min(last,(int) angles.size()-1);
			for(size_t i=last_min_angle;i <= last_max_angle; i++){
				U += angles[i]->energy();
			}
		} else {
			last_min_angle = -1;
			last_max_angle = -1;
		}

		if(print){std::cout << "Energy after second: " << U << std::endl;
		std::cout << "last_min_angle: " << last_min_angle << std::endl;
		std::cout << "last_max_angle: " << last_max_angle << std::endl;
		}
		//Iterate over dihedrals
		//first
		if(first > 0){
			 min_dih = std::max(0,first-3);
			 max_dih = std::min(first-1,((int) dihedrals.size())-1);
			for(size_t i=min_dih;i<=max_dih;i++){
				U += dihedrals[i]->energy();
			}
		} else {
			min_dih = -1;
			max_dih = -1;
		}
		if(print){std::cout << "Energy after third: " << U << std::endl;
		std::cout << "min_dih: " << min_dih << std::endl;
		std::cout << "max_dih: " << max_dih << std::endl;
		}

		//last
		if(last < nbeads-1){
			 last_min_dih = std::max(last-2, (int) max_dih+1);
			 last_max_dih = std::min(last,((int) dihedrals.size())-1);
			for(size_t i=last_min_dih;i<=last_max_dih;i++){
				U += dihedrals[i]->energy();
			}
		} else {
			last_min_dih = -1;
			last_max_dih = -1;
		}
		if(print){std::cout << "Energy after fourth: " << U << std::endl;
		std::cout << "last_min_dih: " << last_min_dih << std::endl;
		std::cout << "last_max_dih: " << last_max_dih << std::endl;
		}

		/*
		if(last-first < 4){
			//Give up on clever indexing and just do them all
			if(first == 1){ U += angles[0]->energy() + dihedrals[0]->energy();}
			else if(first == 2){U += angles}

			size_t angle_first = 
			size_t min_inds = (first-3 < 0) ? 0 : first-3;
			size_t max_inds = (last < dihedrals.size()) ? last : dihedrals.size()-1;
			for(size_t i=min_inds;i <= max_inds; i++){
				U += angles[i]->energy();
				U += dihedrals[i]->energy();
			}
			if(last == nbeads-1) { U += angles[angles.size()-1]->energy();}
			return U;
		}
		

		if (first>0) {
			if(bonds.size() > first) {U += bonds[first-1]->energy();} // move affects bond going into first
			if(angles.size() >= first && first > 1) {U += angles[first-2]->energy() + angles[first-1]->energy();}
			if(dihedrals.size() >= first && first > 2) {U += dihedrals[first-3]->energy() + 
				dihedrals[first-2]->energy() + dihedrals[first-1]->energy();}
			//if only smaller number of dihedrals valid, do those
			if(first == 2){ U += dihedrals[first-2]->energy() + dihedrals[first-1]->energy();}
			if(first == 1){ U += angles[first-1] + dihedrals[first-1]->energy();}
		}

		if (last<(nbeads-1)) {
			if(bonds.size() > last) {U += bonds[last]->energy();} // move affects bond going into first
			if(angles.size() > last && last > 0) {U += angles[last-1]->energy() + angles[last]->energy();}
			if(dihedrals.size() > last && last > 1) {U += dihedrals[last-2]->energy() + 
				dihedrals[last-1]->energy() + dihedrals[last]->energy();}
			//If only one dihedral valid, do that one
			if(last == dihedrals.size()) { U += dihedrals[last-2]->energy() + dihedrals[last-1]->energy();}
			if(last == dihedrals.size()+1) { U += dihedrals[last-2]->energy();}
		}
		*/
		return U;
	}

	double getNonBondedEnergy(const std::unordered_set<Cell*>& flagged_cells)
	{
		// gets all the nonbonded energy
		auto start = std::chrono::high_resolution_clock::now();

		double U = grid.energy(flagged_cells, chis,ideal_chrom);

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
		//std::cout << "NonBonded took " << duration.count() << "microseseconds "<< std::endl;
		return U;
	}

	double getBindingEnergy(const std::unordered_set<Cell*>& flagged_cells)
	{
		// JUST CALCULATES THE VOLUME INTERACTION OF HP1, NOT ENERGY OF SPECIFIC TAIL BINDING
		auto start = std::chrono::high_resolution_clock::now();
		
		double Energy = grid.tailEnergy(flagged_cells);

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
		//std::cout << "Binding took " << duration.count() << "microseseconds "<< std::endl;
	
		return Energy; 
	}


	double getTotalEnergy(int first, int last, const std::unordered_set<Cell*>& flagged_cells)
	{
		double U = 0;
		if (bonded_on) U += getBondedEnergy(first, last);
		//Add ideal chromatin model if enabled in sim
		if (nonbonded_on) U += getNonBondedEnergy(flagged_cells);
		if (binding_on) U += getBindingEnergy(flagged_cells);
		//no LH energy because it's noncooperative
		return U;
	}

	double randomExp(double mu, double decay)
	{
		// generates random number distributed according to two-sided exponential distribution
		// centered about mu, with characteristic decay length
		double cdf_y;
		do {
			cdf_y = rng->uniform();
		} while (cdf_y <= 0); // cdf_y cannot exactly equal 0, otherwise inverse cdf is -infty 


		if (cdf_y > 0.5) {
			return mu - decay*log(1 - 2*abs(cdf_y - 0.5)); // inverse cdf
		}
		else {
			return mu + decay*log(1 - 2*abs(cdf_y - 0.5)); // inverse cdf
		}
	}

	void MC()
	{
		std::cout << "Beginning Simulation" << std::endl;
		double prev_eng, after_eng;

		for(int sweep = 0; sweep<nSweeps; sweep++)
		{
			//std::cout << sweep << std::endl; 
			double nonbonded;

		    //looping:
			//prev_eng = getAllBondedEnergy();

			Timer t_translation("translating", print_MC);
			for(int j=0; j<n_trans; j++)
			{
				MCmove_translate();
				//nonbonded = getNonBondedEnergy(grid.active_cells);
				//std::cout << nonbonded << std::endl;
			}
			//t_translation.~Timer();
			/*
			after_eng = getAllBondedEnergy();
			if(after_eng > prev_eng+1000000){
				std::cout << "Dramatic increase after translation" << std::endl;
			}
			prev_eng = after_eng;
			*/
			if (gridmove_on) MCmove_grid();
			//nonbonded = getNonBondedEnergy(grid.active_cells);
			//std::cout << nonbonded << std::endl;

			Timer t_displace("displacing", print_MC);
			for(int j=0; j<n_disp; j++)
			{
				MCmove_displace();
				//nonbonded = getNonBondedEnergy(grid.active_cells);
				//std::cout << nonbonded << std::endl;
			}
			/*
			//t_displace.~Timer();
			after_eng = getAllBondedEnergy();
			if(after_eng > prev_eng+1000000){
				std::cout << "Dramatic increase after displacement" << std::endl;
			}
			prev_eng = after_eng;
			*/


			Timer t_crankshaft("Cranking", print_MC);
			for(int j=0; j<n_crank; j++) {
				MCmove_crankshaft();
				//nonbonded = getNonBondedEnergy(grid.active_cells);
				//std::cout << nonbonded << std::endl;
			}
			//t_crankshaft.~Timer();
			/*
			after_eng = getAllBondedEnergy();
			if(after_eng > prev_eng+1000000){
				std::cout << "Dramatic increase after crankshaft" << std::endl;
			}
			prev_eng = after_eng;
			*/
			//For fiber sims, remove the rotation moves.
			
			Timer t_rotation("Rotating", print_MC);
			for(int j=0; j<n_rot; j++) {
				MCmove_rotate();
			}
			//t_rotation.~Timer();
			/*
			after_eng = getAllBondedEnergy();
			if(after_eng > prev_eng+1000000){
				std::cout << "Dramatic increase after Rotation" << std::endl;
			}
			prev_eng = after_eng;
			*/

			Timer t_pivot("pivoting", print_MC);
			for(int j=0; j<n_pivot; j++) {
				MCmove_pivot(sweep);
				//nonbonded = getNonBondedEnergy(grid.active_cells);
				//std::cout << nonbonded << std::endl;
			}
			//t_pivot.~Timer();
			/*
			after_eng = getAllBondedEnergy();
			if(after_eng > prev_eng+1000000){
				std::cout << "Dramatic increase after Pivoting" << std::endl;
			}
			prev_eng = after_eng;
			*/

			Timer t_bind("Binding", print_MC);
			for(int j=0; j<n_bind; j++) {
				MCmove_bind();
			}

			correct_hp1();

			if(lh_on){
				for(int j=0; j<n_link; j++) {
					MCmove_link();
				}
			}
			//t_bind.~Timer();
			/*
			after_eng = getAllBondedEnergy();
			if(after_eng > prev_eng+1000000){
				std::cout << "Dramatic increase after binding" << std::endl;
			}
			prev_eng = after_eng;
			*/

			if (sweep%dump_frequency == 0) {
				std::cout << "Sweep number " << sweep << std::endl;
				dumpData();
				
				if (print_acceptance_rates) {
					std::cout << "acceptance rate: " << (float) acc/((sweep+1)*nSteps)*100.0 << "%" << std::endl;
					std::cout << "trans: " << (float) acc_trans/((sweep+1)*n_trans)*100 << "% \t";
					std::cout << "crank: " << (float) acc_crank/((sweep+1)*n_crank)*100 << "% \t";
					std::cout << "pivot: " << (float) acc_pivot/((sweep+1)*n_pivot)*100 << "% \t";
					std::cout << "bind: " << (float) acc_bind/((sweep+1)*n_bind)*100 << "% \t";
					std::cout << "displace: " << (float) acc_disp/((sweep+1)*n_disp)*100 << "% \t";
					std::cout << "link: " << (float) acc_link/((sweep+1)*n_disp)*100 << "% \t";
					std::cout << std::endl;
					//std::cout << "rot: " << (float) acc_rot/((sweep+1)*n_rot)*100 << "%" << std::endl;;
				}

				if (production) {dumpContacts();}
			}

			if (sweep%dump_stats_frequency == 0)
			{
				if (production)
				{
					updateContacts();  // calculate contact data, but dont dump to file
					dumpObservables(sweep);
				}

				Timer t_allenergy("all energy", print_MC);
				double bonded = getAllBondedEnergy();

				double nonbonded = 0;
				double binding = 0;
				nonbonded = nonbonded_on ? getNonBondedEnergy(grid.active_cells) : 0;
				binding = binding_on ? getBindingEnergy(grid.active_cells) : 0;
				//lh_binding = lh_on ? getLHEnergy() : 0;
				//std::cout << "binding " << bonded << " nonbonded " << nonbonded << " binding" << binding <<  std::endl;
				//t_allenergy.~Timer();

				dumpEnergy(sweep, bonded, nonbonded, binding);
			}

		}

		std::cout << "acceptance rate: " << (float) acc/(nSweeps*nSteps)*100.0 << "%" << std::endl;
	}

	
	void MCmove_displace()
	{
		Timer t_displacemove("Displacement move", print_MC);
		// pick random particle
		int o = floor(beads.size()*rng->uniform());

		// copy old info (don't forget orientation, etc)
		Cell* old_cell = grid.getCell(beads[o]);
		
		Eigen::RowVector3d displacement;
		displacement = rng->uniform()*step_disp*unit_vec(displacement);

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
		//double prev_eng = getAllBondedEnergy();
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
			/*
			double post_eng = getAllBondedEnergy();
			if(post_eng > prev_eng){
				std::cout << "Energy failure at bead " << o << ", overall bonded change: ";
				std::cout << post_eng - prev_eng << std::endl;
				double new_bonded = getBondedEnergy(o,o,true);
				std::cout << "Current getBondedEnergy: " << new_bonded << std::endl;
				beads[o].r -= displacement;
				double old_bonded = getBondedEnergy(o,o,true);
				std::cout << "Old getBondedEnergy: " << old_bonded << std::endl;
				std::cout << "getBondedEnergy delta: " << new_bonded - old_bonded << std::endl;
				beads[o].r += displacement;
			}
			*/

		}
		else
		{
			//std::cout << "Rejected" << std::endl;
			beads[o].r -= displacement ;
			new_cell->moveOut(&beads[o]);
			old_cell->moveIn(&beads[o]);
		}
	}

	void MCmove_translate()
	{
		if (print_trans) std::cout << "==================NEW MOVE ====================" << std::endl;
		Timer t_trans("Translate", print_trans);
		Timer t_setup("setup", print_trans);

		// select a first bead at random
		int first = floor(beads.size()*rng->uniform());

		// choose second bead from two-sided exponential distribution around first
		int last = -1; 
		while (last < 0 || last >= nbeads)
		{
			last = round(randomExp(first, exp_decay));            // does this obey detailed balance?
		}

		// swap first and last to ensure last > first
		if (last < first) {std::swap(first, last);} 
		
		if (print_trans) std::cout << "number of beads is " << last - first << std::endl;

		// generate displacement vector with magnitude step_trans
		Eigen::RowVector3d displacement;

		//Add random chance for smaller step
		displacement = rng->uniform()*step_trans*unit_vec(displacement);

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
			Timer t_bounds("bounds", print_trans);
			// reject immediately if moved out of simulation box, no cleanup necessary
			for(int i=first; i<=last; i++)
			{
				new_loc = beads[i].r + displacement;
				if (outside_boundary(new_loc)) {
					throw "exited simulation box";	
				}
			}
			//t_bounds.~Timer();

			Timer t_flag("Flag cells", print_trans);
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
			//double real_Uold = getAllBondedEnergy();
			Timer t_uold("Uold", print_trans);
			//std::cout << "Beads: " << last-first << " Cells: " << flagged_cells.size() << std::endl;
			double Uold = getTotalEnergy(first, last, flagged_cells);
			//t_uold.~Timer();

			Timer t_disp("Displacement", print_trans);
			for(int i=first; i<=last; i++)
			{
				beads[i].r += displacement;
			}
			//t_disp.~Timer();

			Timer t_swap("Bead Swaps", print_trans);
			// update grid <bead index,   <old cell , new cell>>
			//for(std::pair<int, std::pair<Cell*, Cell*>> &x : bead_swaps)
			for(auto const &x : bead_swaps)
			{
				x.second.first->moveOut(&beads[x.first]);
				x.second.second->moveIn(&beads[x.first]);
			}
			//t_swap.~Timer();

			Timer t_unew("Unew", print_trans);
			double Unew = getTotalEnergy(first, last, flagged_cells);
			//t_unew.~Timer();

			if (rng->uniform() < exp(Uold-Unew))
			{
				acc += 1;
				acc_trans += 1;
				nbeads_moved += (last-first);
				/*
				double real_Unew = getAllBondedEnergy();
				if(real_Unew > real_Uold){
					std::cout << "Energy failure at beads " << first << ", " << last;
					std::cout <<", overall bonded change: ";
					std::cout << real_Unew - real_Uold << std::endl;
					double new_bonded = getBondedEnergy(first,last,true);
					std::cout << "Current getBondedEnergy: " << new_bonded << std::endl;
					for(int i=first; i<=last; i++)
					{
						beads[i].r -= displacement;
					}
					double old_bonded = getBondedEnergy(first,last,true);
					std::cout << "Old getBondedEnergy: " << old_bonded << std::endl;
					std::cout << "getBondedEnergy delta: " << new_bonded - old_bonded << std::endl;
					for(int i=first; i<=last; i++)
					{
						beads[i].r += displacement;
					}
				}
				*/
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

	void MCmove_crankshaft()
	{
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
			last = round(randomExp(first, exp_decay_crank));
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
			if (binding_on) Uold += getBindingEnergy(flagged_cells);

			
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

	void MCmove_rotate()
	{
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

	void MCmove_pivot(int sweep)
	{
		// index terminology:
		// pivot === the bead being pivoted around, but not itself moved
		// [first,last] === the interval of beads physically moved in the pivot
	        // depending on which end the pivot is executed, this is either [0, pivot-1] or [pivot+1, nbeads-1]
		// The only bonds affected are pivot-1 OR pivot;

		// chose one end of the polymer and a pivot bead
		int end = (nbeads-1)*round(rng->uniform()); //  either first bead or last bead

		// pick second bead according to single-sided exponential distribution away from end
		int length;
		do {
			length = abs(round(randomExp(0, exp_decay_pivot))); // length down the polymer from the end
		} while (length < 2 || length > nbeads-2);

		int pivot = (end == 0) ? length : (nbeads-1-length);

		int first = (pivot < end) ? pivot+1 : end;
		int last = (pivot < end) ? end : pivot-1;
		//if(first < 10){std::cout << "end: " << end << ", first: " << first << ", last: " << last <<std::endl; 
		//std::cout << "length: " << length << std::endl; throw "actually end =0";}
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
			//if(bonded_on) Uold += getBondedEnergy(pivot-1, pivot);
			double old_bonded = getBondedEnergy(first,last,false);
			if(bonded_on) Uold += old_bonded;

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
			if(binding_on) Uold += getBindingEnergy(flagged_cells);
			
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

	void MCmove_bind()
	{
		// pick random particle

		int o = floor(beads.size()*rng->uniform());

		bool flip2 = round(rng->uniform());       // flip 1 or 2 tails?
		bool flip_first = round(rng->uniform());  // if 1, which tail to flip? 

		Cell* current_cell = grid.getCell(beads[o]); 
		int old_hp1 = hp1_free;
		double Uold = beads[o].tailEnergy(getChemPot()) + current_cell->getTailEnergy();  

		if (flip2) {
			beads[o].tail1.flipBind(current_cell);
			beads[o].tail2.flipBind(current_cell);
		}
		else {  // only flip one tail
			(flip_first) ? beads[o].tail1.flipBind(current_cell) : beads[o].tail2.flipBind(current_cell);
		}
		int new_hp1 = hp1_free;
		double Unew = beads[o].tailEnergy(getChemPot()) + current_cell->getTailEnergy();

		//Reject on hp1 basis of too many hp1s are being consumed
		if(hp1_free < 3 && new_hp1 < old_hp1){
			//std::cout << "Rejected" << std::endl;
			if (flip2) {
				beads[o].tail1.flipBind(current_cell);
				beads[o].tail2.flipBind(current_cell);
			}
			else {
				(flip_first) ? beads[o].tail1.flipBind(current_cell) : beads[o].tail2.flipBind(current_cell);
			}
		}
		else if (rng->uniform() < exp(Uold-Unew))
		{
			//std::cout << "Accepted"<< std::endl;
			acc += 1;
			acc_bind += 1;
		}
		else
		{
			//std::cout << "Rejected" << std::endl;
			
			if (flip2) {
				beads[o].tail1.flipBind(current_cell);
				beads[o].tail2.flipBind(current_cell);
			}
			else {
				(flip_first) ? beads[o].tail1.flipBind(current_cell) : beads[o].tail2.flipBind(current_cell);
			}
		}
	}

	//Add or remove a linker histone to the given particle
	void MCmove_link()
	{
		// pick random particle
		int o = floor(beads.size()*rng->uniform());

		//bool flip2 = round(rng->uniform());       // flip 1 or 2 tails?
		//bool flip_first = round(rng->uniform());  // if 1, which tail to flip? 
		
		//LH binding energy is non-cooperative, based on triplet angles:
		double Uold = beads[o].LHEnergy() + getBondedEnergy(o,o) + getLHChemPot();

		//double Uold = beads[o].tailEnergy(getLHChemPot()) + current_cell->getTailEnergy();  

		int old_lh = lh_free;
		beads[o].flipLH();
		int new_lh = lh_free;
		double Unew = beads[o].LHEnergy() + getBondedEnergy(o,o) + getLHChemPot();

		if(lh_free < 3 && new_lh < old_lh){
			beads[0].flipLH();
		}
		else if (rng->uniform() < exp(Uold-Unew))
		{
			//std::cout << "Accepted"<< std::endl;
			acc += 1;
			acc_link += 1;
		}
		else
		{
			//std::cout << "Rejected" << std::endl;
			beads[0].flipLH();
		}
	}

	void MCmove_grid()
	{
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

			if (U < 9999999999)
			{ 
				flag = false;
				//std::cout << "passed grid move" << std::endl;
			}
			else
			{
				flag = false;
				grid.origin = old_origin;
				grid.meshBeads(beads);
				//std::cout << "failed grid move" << std::endl;
			}
		}
	}

	
	void printTailComposition()
	{
		double HP1_un = 0;
		double HP1_on = 0;
		double HP1_di = 0;

		double HP1_un_tot = 0;
		double HP1_on_tot = 0;
		double HP1_di_tot = 0;

		if (!load_chipseq)
		{
			HP1_un_tot = nbeads/3*2;
			HP1_on_tot = nbeads/3*2;
			HP1_di_tot = nbeads/3*2;
		}
		
		for(Bead bead : beads)
		{
			if (bead.tails_methylated == 0) {
				HP1_un += bead.nbound();
			}
			else if (bead.tails_methylated == 1) {
				HP1_on += bead.nbound();
			}
			else { HP1_di += bead.nbound();}
		}

		//std::cout << "free HP1 is : " << HP1_free << std::endl; // need to modify for Sim::HP1_free
		std::cout << "none: " << HP1_un/HP1_un_tot << " me3: " << HP1_on/HP1_on_tot << " 2xme3: " << HP1_di/HP1_di_tot << std::endl;
	}

	void dumpData() 
	{ 
		xyz_out = fopen("./data_out/output.xyz", "a");
		fprintf(xyz_out, "%d\n", nbeads);
		fprintf(xyz_out, "atoms\n");

		for(Bead bead : beads)
		{
		    //int epimark = bead.tail1.mark + bead.tail2.mark;
		    //int bindmark = bead.tail1.bound + bead.tail2.bound;
		    //fprintf(xyz_out, "%d\t %lf\t %lf\t %lf\t %d\t %d\t %d\n" , bead.id,bead.r(0),bead.r(1),bead.r(2), epimark, bead.nbound(), bead.d);

			fprintf(xyz_out, "%d\t %lf\t %lf\t %lf\t" , bead.id,bead.r(0),bead.r(1),bead.r(2));

			for(int i=0; i<nspecies; i++)
			{
				fprintf(xyz_out, "%d\t", bead.d[i]);
			}

			if(lh_on){
				fprintf(xyz_out,"%d", (int) bead.lh_bound);
			}

			fprintf(xyz_out, "\n");
		}
		fclose(xyz_out); 
	}

	void dumpEnergy(int sweep, double bonded=0, double nonbonded=0, double binding=0)
	{
		energy_out = fopen("./data_out/energy.traj", "a");
		fprintf(energy_out, "%d\t %lf\t %lf\t %lf\n", sweep, bonded, nonbonded, binding);
		fclose(energy_out);
	}

	void dumpObservables(int sweep)
	{
		obs_out = fopen("./data_out/observables.traj", "a");
		fprintf(obs_out, "%d\t", sweep);

		for (int i=0; i<nspecies; i++)
		{
			for (int j=i; j<nspecies; j++)
			{
				double ij_contacts = grid.get_ij_Contacts(i, j);
				fprintf(obs_out, "%lf\t", ij_contacts);
			}
		}

		fprintf(obs_out, "\n");
		fclose(obs_out);
	}

	void dumpContacts()
	{
		// overwrites contact file with most current values
		std::ofstream contactsOutFile("./data_out/contacts.txt");
		for (const auto &row : contact_map) {
			for (const int &element : row) {
				contactsOutFile << element << " ";
			}
			contactsOutFile << "\n";
		}
	}

	void run()
	{
		readInput();
		makeOutputFiles();
		initialize(); 
		grid.generate();  // creates the grid locations
		grid.meshBeads(beads);  // populates the grid locations with beads;
		grid.setActiveCells();  // populates the active cell locations
		setupContacts();
		MC(); // MC simulation
		dumpContacts();
		assert (grid.checkCellConsistency(nbeads));
		assert (grid.checkHp1Consistency(hp1_total));
	}
};

int Sim::hp1_free;
int Sim::lh_free;
int Bead::ntypes;
int Cell::ntypes;

void Bead::flipLH()
{
	lh_bound = !lh_bound;
	if(lh_bound){
		Sim::lh_free -= 1;
	}
	else {
		Sim::lh_free += 1;
	}
}


void Tail::flipBind(Cell* cell_inside)
{
	// update HP1 globally and locally in the cell
	if (bound)
	{
		cell_inside->local_HP1 -= 1;
		Sim::hp1_free += 1;
	}
	else {
		cell_inside->local_HP1 += 1;
		Sim::hp1_free -= 1;
	}

	// flip the binding state
	bound = !bound;
}

bool Grid::checkHp1Consistency(int hp1_tot)
{
	// checks to see if the number of HP1 proteins is conserved
	double cellhp1 = 0;
	for(Cell* cell : active_cells)
	{
		cellhp1 += cell->local_HP1;
	}

	cellhp1 += Sim::hp1_free;
	return (cellhp1 == hp1_tot);
}

int main()
{
	auto start = std::chrono::high_resolution_clock::now();

	std::cout << "Making lookuptable" << std::endl;
	LookupTable3D a(167);
	LookupTable2D b(167);
	LookupTable1D c(167);
	std::cout << a.energy(150.11,148,150.11) << std::endl;
	std::cout << b.energy(150.11,150.11) << std::endl;
	//should be: 6.58908421, 2.78657891, 4.9675081
	
	std::cout << c.energy(159.0) << std::endl;

	Sim mySim;
	mySim.run();

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
	std::cout << "Took " << duration.count() << "seconds "<< std::endl;
	std::cout << "Moved " << nbeads_moved << " beads " << std::endl;
	return 0;
}
