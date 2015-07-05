/* main.cpp
 
 Author        : James Mnatzaganian
 Date Created  : 11/07/14
 
 Description   : Solve the TSP problem using a GA
 License       : MIT License http://opensource.org/licenses/mit-license.php
 Copyright     : (c) 2014 James Mnatzaganian
 */
//  Copyright (c) 2015 waz

// Native Includes
#include <iostream>
#include <string>
#include <ctime>
#include <cstring>

// Program Includes
#include "common.h"
#include "log.h"
#include "ga_cpu.h"
#include "ga_gpu.h"

using namespace std;

int main(int argc, const char * argv[]) {
	// Logger
	Logger gen_log;
	
	// GA parameters
	float prob_mutation  = 0.15f; // The probability of a mutation
	float prob_crossover = 0.8f;  // The probability of a crossover
	int world_seed       = 12345678;    // Seed for initial city selection
	int ga_seed          = 87654321;    // Seed for all other random numbers
	
	// World parameters
	const int world_width  = 10000; // Width of the world
	const int world_height = 10000; // Height of the world
	
	// The test cases
	const int iterations    = 1; // Number of full runs
	const int num_cases     = 1; // How many trials to test
	int cases[num_cases][3] =    // num_cities, pop_size, max_gen
	{
//		{25, 100,    1000},
//		{25, 1000,   1000},
//		{25, 10000,  100},
		{25, 100000, 10}
//		{50, 100,    1000},
//		{50, 1000,   1000},
//		{50, 10000,  100},
//		{50, 100000, 10},
//		
//		{100, 100,    1000},
//		{100, 1000,   1000},
//		{100, 10000,  100},
//		{100, 100000, 10},
//		
//		{250, 100,    1000},
//		{250, 1000,   1000},
//		{250, 10000,  100},
//		{250, 100000, 10}
	};
	
	// Timing
	clock_t iter_time, total_time;
	
	// The output path
	std::string path = "./";
	
	// Loop over all city combinations
	for (int i=0; i<num_cases; i++)
	{
		// GA params
		int num_cities = cases[i][0];
		int pop_size   = cases[i][1];
		int max_gen    = cases[i][2];
		
		// Generate new strings for the output
		std::string c_timing_path;
		std::string c_gen_path;
		std::string c_stats_path;
		std::string g_timing_path;
		std::string g_gen_path;
		std::string g_stats_path;
		
		std::string c0, c1, c2;
		std::string g0, g1, g2;
		
		c0 = to_string(num_cities) + "_" + to_string(pop_size) + "-cpu_gen.csv";
		c1 = to_string(num_cities) + "_" + to_string(pop_size) + "-cpu_timing.csv";
		c2 = to_string(num_cities) + "_" + to_string(pop_size) + "-cpu_stats.csv";
		g0 = to_string(num_cities) + "_" + to_string(pop_size) + "-gpu_gen.csv";
		g1 = to_string(num_cities) + "_" + to_string(pop_size) + "-gpu_timing.csv";
		g2 = to_string(num_cities) + "_" + to_string(pop_size) + "-gpu_stats.csv";
		
		// Make the world
		World world(num_cities, world_height, world_width, world_seed);
		
		// Build the strings for logging purposes
		c_timing_path = path + c0;
		c_gen_path = path + c1;
		c_stats_path = path + c2;
		g_timing_path = path + g0;
		g_gen_path = path + g1;
		g_stats_path = path + g2;
		
		cout << endl;
		cout << "###############################################################################" << endl;
		cout << "##### CPU - START" << endl;
		cout << "###############################################################################" << endl << endl;
		
		// CPU timing
		gen_log.start(c_gen_path, c_timing_path, c_stats_path);
		total_time = clock();
		for (int j=0; j<iterations; j++)
		{
			iter_time = clock();
			execute(pop_size, max_gen, prob_mutation, prob_crossover, world, gen_log, ga_seed);
			gen_log.write_stats(j + 1, "CPU", end_clock(iter_time),
								 prob_mutation, prob_crossover, pop_size, max_gen, world_seed,
								 ga_seed, world_width, world_height, num_cities);
		}
		gen_log.write_stats(-1, "CPU", end_clock(total_time),
							 prob_mutation, prob_crossover, pop_size, max_gen, world_seed,
							 ga_seed, world_width, world_height, num_cities);
		gen_log.end();
		
		cout << endl;
		cout << "###############################################################################" << endl;
		cout << "CPU - END" << endl;
		cout << "###############################################################################" << endl << endl;
		
		cout << "===============================================================================" << endl << endl;
		
		cout << "###############################################################################" << endl;
		cout << "GPU - START" << endl;
		cout << "###############################################################################" << endl << endl;
		
		// GPU warmup pass - A single generation should be good enough
		gen_log.start(g_gen_path, g_timing_path, g_stats_path);
		g_execute(pop_size, 1, prob_mutation, prob_crossover, world, gen_log, ga_seed);
		gen_log.end();
		
		// GPU timing
		gen_log.start(g_gen_path, g_timing_path, g_stats_path);
		total_time = clock();
		for (int j=0; j<iterations; j++)
		{
			iter_time = clock();
			g_execute(pop_size, max_gen, prob_mutation, prob_crossover, world, gen_log, ga_seed);
			gen_log.write_stats(j + 1, "GPU", end_clock(iter_time),
								prob_mutation, prob_crossover, pop_size, max_gen, world_seed,
								ga_seed, world_width, world_height, num_cities);
		}
		gen_log.write_stats(-1, "GPU", end_clock(total_time), prob_mutation,
							 prob_crossover, pop_size, max_gen, world_seed, ga_seed,
							 world_width, world_height, num_cities);
		gen_log.end();
		
		cout << endl;
		cout << "###############################################################################" << endl;
		cout << "GPU - END" << endl;
		cout << "###############################################################################" << endl << endl;
	}
	
    return 0;
}
