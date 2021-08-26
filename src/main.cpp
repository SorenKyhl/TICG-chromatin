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
#include "random_mars.h"
#include "nlohmann/json.hpp"
#include "prof_timer.cpp"

#include "Bead.h"
#include "Bond.h"
#include "DSS_Bond.h"
#include "Cell.cpp"
#include "Grid.h"
#include "Sim.h"

int main()
{
	auto start = std::chrono::high_resolution_clock::now();

	Sim mySim;
	mySim.run();

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
	std::cout << "Took " << duration.count() << "seconds "<< std::endl;
	std::cout << "Moved " << mySim.nbeads_moved << " beads " << std::endl;
	return 0;
}
