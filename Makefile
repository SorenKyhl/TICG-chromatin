CC = g++
FLAGS = -O3
DEBUG_FLAGS = -g

all: 
	$(CC) $(FLAGS) main.cpp prof_timer.cpp

debug: 
	$(CC) $(DEBUG_FLAGGS) main.cpp prof_timer.cpp
