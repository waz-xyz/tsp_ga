/* ga_gpu.cu
 
 Author        : James Mnatzaganian
 Date Created  : 11/07/14
 
 Description   : GA implementation for the GPU
 License       : MIT License http://opensource.org/licenses/mit-license.php
 Copyright     : (c) 2014 James Mnatzaganian
 */
// Copyright (c) 2015 waz

// Native includes
#include <iostream>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <cstring>
#include <cassert>

// Program includes
#include "g_type.h"
#include "g_population.h"
#include "ga_gpu.h"
#include "common.h"
#include "log.h"

void g_execute(int pop_size,
			   int max_gen,
			   float prob_mutation, float prob_crossover,
			   const World& baseWorld,
			   Logger& gen_log,
			   int seed)
{
	// Timing
	clock_t gen_clock;
	
	// Random number generation
	std::mt19937::result_type rseed = seed;
	auto rgen = bind(uniform_real_distribution<float>(0, 1), mt19937(rseed));
	
	// The populations
	g_Population* old_pop;
	g_Population* new_pop;
	
	// Random numbers
	float *prob_select = new float[2 * pop_size];
	float *prob_cross  = new float[pop_size];
	float *prob_mutate = new float[pop_size];
	int   *cross_loc   = new int[pop_size];
	int   *mutate_loc  = new int[2 * pop_size];
	
	// Best individual parameters
	int   sel;
	int   best_generation = 0;
	World best_leader(baseWorld.num_cities, baseWorld.height, baseWorld.width);
	World generation_leader(baseWorld.num_cities, baseWorld.height, baseWorld.width);
	
	///////// CPU Initializations
	opencl_env env;
	
	///////// GPU Allocations
	// Populations
	old_pop = new g_Population(env, pop_size, baseWorld, seed);
	new_pop = new g_Population(env, pop_size, baseWorld.num_cities, baseWorld.height, baseWorld.width);
	
	// Random numbers
	cl::Buffer d_prob_select = cl::Buffer(env.context(), CL_MEM_READ_ONLY, sizeof(float) * 2 * pop_size);
	cl::Buffer d_prob_cross = cl::Buffer(env.context(), CL_MEM_READ_ONLY, sizeof(float) * pop_size);
	cl::Buffer d_prob_mutate = cl::Buffer(env.context(), CL_MEM_READ_ONLY, sizeof(float) * pop_size);
	cl::Buffer d_cross_loc = cl::Buffer(env.context(), CL_MEM_READ_ONLY, sizeof(int) * pop_size);
	cl::Buffer d_mutate_loc = cl::Buffer(env.context(), CL_MEM_READ_ONLY, sizeof(int) * 2 * pop_size);

	// Other parameters
	cl::Buffer d_sel_ix = cl::Buffer(env.context(), CL_MEM_READ_WRITE, sizeof(int) * 2 * pop_size);
	
	///////// GPU Initializations
	
	// Calculate the fitnesses
	old_pop->evaluate();
	
	// Initialize the best leader
	old_pop->select_leader(generation_leader, best_leader);
	print_status(generation_leader, best_leader, 0);
	gen_log.write_log(0, 0, generation_leader);
	
	// Continue through all generations
	for (int i = 0; i < max_gen; i++)
	{
		// Start the generation clock
		gen_clock = clock();
		
		// Generate all probabilities for each step
		//
		// The order the random numbers are generated must be consistent to
		// ensure the results will match the CPU.
		for (int j = 0; j < pop_size; j++)
		{
			prob_select[2*j] = rgen();
			prob_select[2*j + 1] = rgen();
			prob_cross[j] = rgen();
			prob_mutate[j] = rgen();
			
			cross_loc[j] = static_cast<int>(rgen() * (baseWorld.num_cities - 1));
			
			int mut_loc_0 = static_cast<int>(rgen() * (baseWorld.num_cities));
			int mut_loc_1 = static_cast<int>(rgen() * (baseWorld.num_cities));
			while (mut_loc_1 == mut_loc_0) {
				mut_loc_1 = static_cast<int>(rgen() * baseWorld.num_cities);
			}
			mutate_loc[2*j]      = mut_loc_0;
			mutate_loc[2*j + 1]  = mut_loc_1;
		}
		
		// Copy random numbers to device
		env.queue().enqueueWriteBuffer(d_prob_select, CL_FALSE, 0, 2*pop_size*sizeof(float), prob_select);
		env.queue().enqueueWriteBuffer(d_prob_cross, CL_FALSE, 0, pop_size*sizeof(float), prob_cross);
		env.queue().enqueueWriteBuffer(d_prob_mutate, CL_FALSE, 0, pop_size*sizeof(float), prob_mutate);
		env.queue().enqueueWriteBuffer(d_cross_loc, CL_FALSE, 0, pop_size*sizeof(int), cross_loc);
		env.queue().enqueueWriteBuffer(d_mutate_loc, CL_FALSE, 0, 2*pop_size*sizeof(int), mutate_loc);
		
		// Select the parents
		old_pop->select_parents(d_sel_ix, d_prob_select);
		
		// Create the children (form the new population entirely on the GPU!)
		old_pop->next_generation(*new_pop,
								 d_sel_ix,
								 prob_crossover, d_prob_cross, d_cross_loc,
								 prob_mutation, d_prob_mutate, d_mutate_loc);
		
		// Calculate the fitnesses on the new population
		new_pop->evaluate();
		
		// Swap the populations
		std::swap(old_pop, new_pop);
		
		// Select the new leaders
		sel = old_pop->select_leader(generation_leader, best_leader);
		if (sel == 1)
		{
			best_generation = i + 1;
		}
		print_status(generation_leader, best_leader, i + 1);
		gen_log.write_log(i + 1, end_clock(gen_clock), generation_leader);
	} // Generations
	
	std::cout
		<< std::endl
		<< "Best generation found at " << best_generation << " generations"
		<< std::endl;
	
	// Cleanup and success!
	delete old_pop;
	delete new_pop;
	delete[] prob_select; delete[] prob_cross; delete[] prob_mutate;
	delete[] cross_loc; delete[] mutate_loc;
}
