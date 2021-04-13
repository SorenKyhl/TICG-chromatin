#pragma once

class Bead {
public:
	Bead(int i, double x=0, double y=0, double z=0, int ntypes=1)
		: id{i}, r{x,y,z}, d(ntypes) {}
	
	Bead() 
		: id{0}, r{0,0,0}, d(1) {}

	int id;   // unique identifier: uniqueness not strictly enforced.
	Eigen::RowVector3d r; // position
	Eigen::RowVector3d u; // orientation

	// number of different bead types
	static int ntypes;
	//std::vector<int> d = std::vector<int>(ntypes);
	std::vector<int> d;

	void print() {std::cout << id <<" "<< r << std::endl;}
};
