CC = g++
FLAGS = -O3
DEBUG_FLAGS = -g
INCLUDE = -I ./include
SRC = main.cpp
OUT = -o TICG-engine
LINK = -L/home/coraor/lib/ -lcnpy -lz -std=c++11

all: 
	$(CC) $(FLAGS) $(INCLUDE) $(SRC) $(LINK) $(OUT)

debug: 
	$(CC) $(DEBUG_FLAGS) $(INCLUDE) $(SRC) $(LINK) $(OUT)
