CC = g++
FLAGS = -O3
DEBUG_FLAGS = -g
INCLUDE = -I ./include
SRC = main.cpp

all: 
	$(CC) $(FLAGS) $(INCLUDE) $(SRC)

debug: 
	$(CC) $(DEBUG_FLAGGS) $(INCLUDE) $(SRC)
