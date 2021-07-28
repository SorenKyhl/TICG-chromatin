CC = g++
FLAGS = -O3 -std=c++14
DEBUG_FLAGS = -g
INCLUDE = -I ./include
SRC = main.cpp
OUT = -o TICG-engine

all: 
	$(CC) $(FLAGS) $(INCLUDE) $(SRC) $(OUT)

debug: 
	$(CC) $(DEBUG_FLAGS) $(INCLUDE) $(SRC) $(OUT)
