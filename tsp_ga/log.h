/* log.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Header for all logging
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/
//  Copyright (c) 2015 waz

#ifndef __LOG_H__
#define __LOG_H__

// Native includes
#include <fstream>
#include <cassert>

// Program includes
#include "world.h"

using namespace std;

class Logger
{
	/*
		Handles all logging operations
	*/

private:
	// File operators
	ofstream timing_data;
	ofstream generation_data;
	ofstream stats_data;

public:
	void start(std::string timing_path, std::string generation_path, std::string stats_path);
	
	void write_log(int generation, float gen_time, const World& leader);
	
	void write_stats(int iteration, const char* type, float total_time,
		float prob_mutation, float prob_crossover, int pop_size, int max_gen, 
		int world_seed, int ga_seed, int world_width, int world_height,       
					 int num_cities);
	
	void end();
};

/*
	Prints the current status to stdout
	
	generation_leader : The leader for the current generation
	best_leader       : The leader out of all generations
	generation        : The generation index
*/
void print_status(const World& generationLeader, const World& bestLeader, int generation);

#endif
