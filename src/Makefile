CC = g++
FLAGS = -O3 -std=c++11
DEBUG_FLAGS = -g -std=c++11
INCLUDE = -I ../include

all: TICG-engine

random_mars.o: random_mars.cpp random_mars.h
	$(CC) $(FLAGS) -c $<

Grid.o: Grid.cpp Grid.h
	$(CC) $(FLAGS) $(INCLUDE) -c $<

Sim.o: Sim.cpp Sim.h
	$(CC) $(FLAGS) $(INCLUDE) -c $<

TICG-engine: main.cpp Grid.o Sim.o random_mars.o
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ 

debug2: main.cpp Grid.o Sim.o random_mars.o
	$(CC) $(DEBUG_FLAGS) $(INCLUDE) -o TICG-engine $^ 

debug: 
	g++ $(DEBUG_FLAGS) -I ../include -c Grid.cpp
	g++ $(DEBUG_FLAGS) -I ../include -c Sim.cpp
	g++ $(DEBUG_FLAGS) -c random_mars.cpp
	g++ $(DEBUG_FLAGS) -I ../include -o TICG-engine main.cpp Grid.o Sim.o random_mars.o
