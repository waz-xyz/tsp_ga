//
//  g_population.cpp
//  tsp_ga
//
//  Created by waz on 20/06/15.
//  Copyright (c) 2015 waz
//

#include "common.h"
#include "g_population.h"

g_Population::g_Population(
	const opencl_env& env,
	int numIndividuals, int numCitiesPerWorld,
	int height, int width)
:
	env(env),
	numIndividuals(numIndividuals), numCitiesPerWorld(numCitiesPerWorld),
	height(height), width(width)
{
	cl::Context& context = env.context();
	
	int totalCities = numCitiesPerWorld * numIndividuals;
	this->cities_xcoord = cl::Buffer(context, CL_MEM_READ_WRITE, totalCities * sizeof(int));
	this->cities_ycoord = cl::Buffer(context, CL_MEM_READ_WRITE, totalCities * sizeof(int));
	this->fitness = cl::Buffer(context, CL_MEM_READ_WRITE, numIndividuals * sizeof(float));
	this->fit_prob = cl::Buffer(context, CL_MEM_READ_WRITE, numIndividuals * sizeof(float));
}

g_Population::g_Population(
	const opencl_env& env,
	int numIndividuals,
	const World& baseWorld,
	int seed)
:
	env(env),
	numIndividuals(numIndividuals), numCitiesPerWorld(baseWorld.num_cities),
	height(baseWorld.height), width(baseWorld.width)
{
	cl::Context& context = env.context();
	
	int totalCities = numCitiesPerWorld * numIndividuals;
	this->cities_xcoord = cl::Buffer(context, CL_MEM_READ_WRITE, totalCities * sizeof(int));
	this->cities_ycoord = cl::Buffer(context, CL_MEM_READ_WRITE, totalCities * sizeof(int));
	this->fitness = cl::Buffer(context, CL_MEM_READ_WRITE, numIndividuals * sizeof(float));
	this->fit_prob = cl::Buffer(context, CL_MEM_READ_WRITE, numIndividuals * sizeof(float));
	
	// Set the seed for random number generation
	srand(seed);
	
	int* x_coord = new int[totalCities];
	int* y_coord = new int[totalCities];
	City* cities = new City[numCitiesPerWorld];
	memmove(cities, baseWorld.cities, numCitiesPerWorld*sizeof(City));
	
	for (int i = 0; i < numIndividuals; i++)
	{
		int baseOffset = i * numCitiesPerWorld;
		random_shuffle(cities, &cities[numCitiesPerWorld]);
		
		for (int j = 0; j < numCitiesPerWorld; j++) {
			x_coord[baseOffset + j] = cities[j].x;
			y_coord[baseOffset + j] = cities[j].y;
		}
	}
	
	env.queue().enqueueWriteBuffer(cities_xcoord, CL_FALSE, 0, totalCities*sizeof(int), x_coord);
	env.queue().enqueueWriteBuffer(cities_ycoord, CL_FALSE, 0, totalCities*sizeof(int), y_coord);
	
	delete[] cities;
	delete[] x_coord;
	delete[] y_coord;
}

void g_Population::evaluate()
{
	cl::Kernel& k_fitness = env.getKernel(kernel_t::fitness);
	cl::Kernel& k_fit_sum = env.getKernel(kernel_t::fit_sum);
	cl::Kernel& k_fit_prob = env.getKernel(kernel_t::fit_prob);
	cl::Buffer d_fit_sum = cl::Buffer(env.context(), CL_MEM_READ_WRITE, sizeof(float));
	cl::NDRange globalws(numIndividuals);
	float fit_sum;
	
	// Calculate the fitnesses
	k_fitness.setArg(0, numIndividuals);
	k_fitness.setArg(1, numCitiesPerWorld);
	k_fitness.setArg(2, width*height);
	k_fitness.setArg(3, cities_xcoord);
	k_fitness.setArg(4, cities_ycoord);
	k_fitness.setArg(5, fitness);
	env.queue().enqueueNDRangeKernel(k_fitness, cl::NullRange, globalws);
	
	// Calculate the total sum and compute the partial probabilities
	k_fit_sum.setArg(0, numIndividuals);
	k_fit_sum.setArg(1, fitness);
	k_fit_sum.setArg(2, fit_prob);
	k_fit_sum.setArg(3, d_fit_sum);
	env.queue().enqueueNDRangeKernel(k_fit_sum, cl::NullRange, globalws);
	env.queue().enqueueReadBuffer(d_fit_sum, CL_TRUE, 0, sizeof(float), &fit_sum);
	
	// Compute the full probabilities
	k_fit_prob.setArg(0, numIndividuals);
	k_fit_prob.setArg(1, fit_prob);
	k_fit_prob.setArg(2, fit_sum);
	env.queue().enqueueNDRangeKernel(k_fit_prob, cl::NullRange, globalws);
}

int g_Population::select_leader(World& generation_leader, World& best_leader) const
{
	/*
		Updates the generation and global best leaders
	 
		generation_leader : The world with the max fitness for this generation
		best_leader       : The world with the best global fitness across all generations
	 
		return 1 if this generation is the best, 0 if not, and -1 for error
	 */
	
	// Calculate the max fitness
	extract_max_fit(generation_leader);
	
	// Update best leader
	if (generation_leader.fitness > best_leader.fitness)
	{
		best_leader = generation_leader;
		return 1;
	}
	
	return 0;
}

void g_Population::extract_max_fit(World& res_world) const
{
	cl::Kernel& k0 = env.getKernel(kernel_t::max_fit_phase_0);
	cl::Kernel& k1 = env.getKernel(kernel_t::max_fit_phase_1);
	
	const int grpSize = 256;
	const int nofGroups = env.getNumComputeUnits()*4;
	cl::Buffer result_val = cl::Buffer(env.context(), CL_MEM_READ_WRITE, nofGroups*sizeof(float));
	cl::Buffer result_inx = cl::Buffer(env.context(), CL_MEM_READ_WRITE, nofGroups*sizeof(int));
	k0.setArg(0, numIndividuals);
	k0.setArg(1, fitness);
	k0.setArg(2, cl::Local(grpSize*sizeof(float)));
	k0.setArg(3, cl::Local(grpSize*sizeof(int)));
	k0.setArg(4, result_val);
	k0.setArg(5, result_inx);
	
	cl::NDRange localws(grpSize);
	cl::NDRange globalws(nofGroups*grpSize);
	env.queue().enqueueNDRangeKernel(k0, cl::NullRange, globalws, localws);
	
	k1.setArg(0, nofGroups);
	k1.setArg(1, result_val);
	k1.setArg(2, result_inx);
	env.queue().enqueueNDRangeKernel(k1, cl::NullRange, cl::NDRange(1));
	
	int max_inx;
	env.queue().enqueueReadBuffer(result_inx, CL_TRUE, 0, sizeof(int), &max_inx);
	
	env.queue().enqueueReadBuffer(fitness, CL_FALSE, max_inx*sizeof(float), sizeof(float), &res_world.fitness);
	env.queue().enqueueReadBuffer(fit_prob, CL_FALSE, max_inx*sizeof(float), sizeof(float), &res_world.fit_prob);
	
	int* xcoord = new int[numCitiesPerWorld];
	int* ycoord = new int[numCitiesPerWorld];
	int base_offset = max_inx * numCitiesPerWorld * sizeof(int);
	env.queue().enqueueReadBuffer(cities_xcoord, CL_FALSE, base_offset, numCitiesPerWorld*sizeof(int), xcoord);
	env.queue().enqueueReadBuffer(cities_ycoord, CL_TRUE, base_offset, numCitiesPerWorld*sizeof(int), ycoord);
	
	for (int i = 0; i < numCitiesPerWorld; i++) {
		res_world.cities[i].x = xcoord[i];
		res_world.cities[i].y = ycoord[i];
	}
	
	delete[] xcoord;
	delete[] ycoord;
}

void g_Population::select_parents(cl::Buffer& selected_inx, cl::Buffer& probs) const
{
	cl::Kernel& k = env.getKernel(kernel_t::select_parents);
	k.setArg(0, 2 * numIndividuals);
	k.setArg(1, fit_prob);
	k.setArg(2, probs);
	k.setArg(3, selected_inx);
	cl::NDRange globalws(2 * numIndividuals);
	env.queue().enqueueNDRangeKernel(k, cl::NullRange, globalws);
}

void g_Population::next_generation(g_Population& new_pop,
								   const cl::Buffer& d_selected_parents_inx,
								   float prob_crossover,
								   const cl::Buffer d_rnd_prob_cross, const cl::Buffer d_cross_loc,
								   float prob_mutation,
								   const cl::Buffer d_rnd_prob_mutate, const cl::Buffer d_mutate_loc) const
{
	cl::Kernel& k_crossover = env.getKernel(kernel_t::crossover);
	cl::Kernel& k_clone_parent = env.getKernel(kernel_t::clone_parent);
	cl::Kernel& k_mutate = env.getKernel(kernel_t::mutate);
	
	cl::NDRange globalws(numIndividuals);
	
//	int pop_len,
//	int num_cities,
//	__global const int* old_x_coord,
//	__global const int* old_y_coord,
//	__global int* new_x_coord,
//	__global int* new_y_coord,
//	__global int* selected_parents_inx,
//	float prob_crossover,
//	__global float* rnd_prob_cross,
//	__global int* cross_loc
	k_crossover.setArg(0, numIndividuals);
	k_crossover.setArg(1, numCitiesPerWorld);
	k_crossover.setArg(2, cities_xcoord);
	k_crossover.setArg(3, cities_ycoord);
	k_crossover.setArg(4, new_pop.cities_xcoord);
	k_crossover.setArg(5, new_pop.cities_ycoord);
	k_crossover.setArg(6, d_selected_parents_inx);
	k_crossover.setArg(7, prob_crossover);
	k_crossover.setArg(8, d_rnd_prob_cross);
	k_crossover.setArg(9, d_cross_loc);
	env.queue().enqueueNDRangeKernel(k_crossover, cl::NullRange, globalws);
	
//	int pop_len,
//	int num_cities,
//	__global const int* old_x_coord,
//	__global const int* old_y_coord,
//	__global int* new_x_coord,
//	__global int* new_y_coord,
//	float prob_crossover,
//	__global const float* rnd_prob_cross,
//	__global const int* selected_parents_inx
	k_clone_parent.setArg(0, numIndividuals);
	k_clone_parent.setArg(1, numCitiesPerWorld);
	k_clone_parent.setArg(2, cities_xcoord);
	k_clone_parent.setArg(3, cities_ycoord);
	k_clone_parent.setArg(4, new_pop.cities_xcoord);
	k_clone_parent.setArg(5, new_pop.cities_ycoord);
	k_clone_parent.setArg(6, prob_crossover);
	k_clone_parent.setArg(7, d_rnd_prob_cross);
	k_clone_parent.setArg(8, d_selected_parents_inx);
	env.queue().enqueueNDRangeKernel(k_clone_parent, cl::NullRange, globalws);
	
//	int pop_len,
//	int num_cities,
//	__global int* x_coord,
//	__global int* y_coord,
//	float prob_mutation,
//	__global const float* rnd_prob_mutation,
//	__global const int* rnd_mutate_loc
	k_mutate.setArg(0, numIndividuals);
	k_mutate.setArg(1, numCitiesPerWorld);
	k_mutate.setArg(2, new_pop.cities_xcoord);
	k_mutate.setArg(3, new_pop.cities_ycoord);
	k_mutate.setArg(4, prob_mutation);
	k_mutate.setArg(5, d_rnd_prob_mutate);
	k_mutate.setArg(6, d_mutate_loc);
	env.queue().enqueueNDRangeKernel(k_mutate, cl::NullRange, globalws);
}
