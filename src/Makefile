CC = g++
FLAGS = -O3 -std=c++11
DEBUG_FLAGS = -g -std=c++11
INCLUDE = -I ../include

all: main.cpp Grid.o Sim.o random_mars.o
	$(CC) $(FLAGS) $(INCLUDE) -o TICG-engine $^ 

random_mars.o: random_mars.cpp random_mars.h
	$(CC) $(FLAGS) -c $<


Grid.o: Grid.cpp Grid.h
	$(CC) $(FLAGS) $(INCLUDE) -c $<

Sim.o: Sim.cpp Sim.h
	$(CC) $(FLAGS) $(INCLUDE) -c $<

debug: 
	$(CC) $(DEBUG_FLAGS) $(INCLUDE) $(SRC) $(OUT)
