CC = g++
FLAGS = -O3 -std=c++17 -fopenmp
DEBUG_FLAGS = -g -std=c++17
INCLUDE = -I ../include
MAIN = main.cpp
EXE = TICG-engine

all: $(EXE)

random_mars.o: random_mars.cpp random_mars.h
	$(CC) $(FLAGS) -c $<

Grid.o: Grid.cpp Grid.h
	$(CC) $(FLAGS) $(INCLUDE) -c $<

Sim.o: Sim.cpp Sim.h
	$(CC) $(FLAGS) $(INCLUDE) -c $<

$(EXE): $(MAIN) Grid.o Sim.o random_mars.o
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ 

debug2: $(MAIN) Grid.o Sim.o random_mars.o
	$(CC) $(DEBUG_FLAGS) $(INCLUDE) -o TICG-engine $^ 

debug: 
	g++ $(DEBUG_FLAGS) -I ../include -c Grid.cpp
	g++ $(DEBUG_FLAGS) -I ../include -c Sim.cpp
	g++ $(DEBUG_FLAGS) -c random_mars.cpp
	g++ $(DEBUG_FLAGS) -I ../include -o TICG-debug $(MAIN) Grid.o Sim.o random_mars.o

pybind:
	g++ -O3 -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) pybind_Sim.cpp -o pyticg$$(python3-config --extension-suffix)

clean:
	rm *.o
