CC = g++
FLAGS = -g -std=c++11
DEBUG_FLAGS = -O3
INCLUDE = -I ./include
SRC = main.cpp
OUT = -o TICG-engine

all: 
	$(CC) $(FLAGS) $(INCLUDE) $(SRC) $(OUT)

debug: 
	$(CC) $(DEBUG_FLAGS) $(INCLUDE) $(SRC) $(OUT)
