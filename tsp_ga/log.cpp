/* log.cpp

   Author        : James Mnatzaganian
   Contact       : http://techtorials.me
   Date Created  : 11/07/14
   
   Description   : Module for handling all logging operations
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/
//  Copyright (c) 2015 waz

// Native includes
#include <iostream>

// Program includes
#include "log.h"

using namespace std;

void Logger::start(std::string timing_path, std::string generation_path, std::string stats_path)
{
	/*
		Starts logging
	 
		timing_path     : The full path to where the timing data should be saved.
		generation_path : The full path to where the generation data should	be saved.
		stats_path      : The full path to where the overall details should	be saved.
	 */
	
	timing_data.open(timing_path);
	generation_data.open(generation_path);
	stats_data.open(stats_path);
	
	assert(timing_data.is_open());
	assert(generation_data.is_open());
	assert(stats_data.is_open());
	
	timing_data << "Generation,Time [ms],Fitness,Distance" << endl;
	stats_data << "Iteration,Type,Total Time [ms],Probability of Mutation,"
	"Probability of Crossover,Population Size,Total Generations,"
	"World Seed,GA Seed,Width of World,"
	"Height of World,Number of Cities" << endl;
}
	
void Logger::write_log(int generation, float gen_time, const World& leader)
{
	/*
		Writes to the log file
	 
		generation : The current generation number
		gen_time   : The execution time for the current generation
		leader     : The leader for the current generation
	 */
	
	// Timing data
	timing_data << generation << "," << gen_time << ","
	<< leader.fitness << "," << leader.calc_distance() << endl;
	
	// Generation data
	for (int i=0; i<leader.num_cities-1; i++) {
		generation_data << leader.cities[i].x << "_" << leader.cities[i].y << ",";
	}
	generation_data << leader.cities[leader.num_cities-1].x << "_"
	<< leader.cities[leader.num_cities-1].y << endl;
}
	
void Logger::write_stats(int iteration, const char* type, float total_time,
						 float prob_mutation, float prob_crossover, int pop_size,
						 int max_gen, int world_seed, int ga_seed, int world_width,
						 int world_height, int num_cities)
{
	/*
		Writes the details of this simulation to a log file
		
		iteration      : The current iteration number
		type           : "CPU" or "GPU" for which data was being measured
		total_time     : The total execution time
		prob_mutation  : The probability of a mutation occurring
		prob_crossover : The probability of a crossover occurring
		pop_size       : The number of elements in the population
		max_gen        : The number of generations to run for
		world_seed     : The seed used to generate the cities
		ga_seed        : The seed used for everything else
		world_width    : The width of the world
		world_height   : The height of the world
		num_cities     : The number of cities in the world
	 */
	
	stats_data << iteration << "," << type << "," << total_time << ","
	<< prob_mutation << "," << prob_crossover << "," << pop_size
	<< "," << max_gen << "," << world_seed << "," << ga_seed << ","
	<< world_width << "," << world_height << "," << num_cities
	<< endl;
}

void Logger::end()
{
	/*
		Closes the log files
	 */
	
	timing_data.close();
	generation_data.close();
	stats_data.close();
}

void print_status(const World& generationLeader, const World& bestLeader, int generation)
{
	/*
		Prints the current status to stdout
		
		generation_leader : The leader for the current generation
		best_leader       : The leader out of all generations
		generation        : The generation index
	*/
	
	cout << "Generation " << generation << ":" << endl;
	cout << "  Generation Leader's Fitness: "  << generationLeader.fitness << endl;
	cout << "  Best Leader's Fitness      : "  << bestLeader.fitness << endl;
}