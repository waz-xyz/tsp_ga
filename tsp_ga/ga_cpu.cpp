/* ga_cpu.cpp

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : GA implementation for the CPU
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/
//  Copyright (c) 2015 waz

// Native includes
#include <iostream>
#include <algorithm>
#include <random>
#include <utility>
#include <cstring>
#include <cassert>

// Program includes
#include "ga_cpu.h"
#include "population.h"
#include "common.h"
#include "log.h"

using namespace std;

void evaluate(Population& pop)
{
	// Sum of all fitness
	float fit_sum = 0.0f;

	// Calculate fitnesses and total sum; compute partial prob
	for (int i = 0; i < pop.numIndividuals; i++)
	{
		fit_sum += pop.CalcFitness(i);
		pop.fit_prob[i] = fit_sum;
	}

	// Compute the full probabilities
	for (int i = 0; i < pop.numIndividuals; i++)
		pop.fit_prob[i] /= fit_sum;
}

void selection(const Population& pop, int* parent_xcoord[], int* parent_ycoord[], float rand_nums[2])
{
	// Select the parents
	for (int i = 0; i < 2; i++) {
		float prob = rand_nums[i];
		for (int j = 0; j < pop.numIndividuals; j++) {
			if (prob <= pop.fit_prob[j]) {
				pop.GetCities(parent_xcoord[i], parent_ycoord[i], j);
				break;
			}
		}
	}
}

void crossover(int* parents_xcoord[2], int* parents_ycoord[2], int* child_xcoord, int* child_ycoord, int num_cities, int cross_over)
{
	// Select elements in first parent from start up through crossover point
	memmove(child_xcoord, parents_xcoord[0], (cross_over + 1) * sizeof(int));
	memmove(child_ycoord, parents_ycoord[0], (cross_over + 1) * sizeof(int));
	
	// Add remaining elements from second parent to child, preserving order
	int* p_xcoord = parents_xcoord[1];
	int* p_ycoord = parents_ycoord[1];
	int remaining = num_cities - cross_over - 1; // The number of cities to add
	int count     = 0; // The number of cities that have been added
	for (int i = 0; i < num_cities; i++) // Loop parent
	{
		bool in_child = false;
		for (int j = 0; j <= cross_over; j++) // Loop child
		{
			// If the city is in the child, exit this loop
			if (child_xcoord[j] == p_xcoord[i] && child_ycoord[j] == p_ycoord[i])
			{
				in_child = true;
				break;
			}
		}
			
		// If the city was not found in the child, add it to the child
		if (!in_child)
		{
			count++;
			child_xcoord[cross_over+count] = p_xcoord[i];
			child_ycoord[cross_over+count] = p_ycoord[i];
		}
			
		// Stop once all of the cities have been added
		if (count == remaining) break;
	}
}

void mutate(int* child_xcoord, int* child_ycoord, int rand_nums[2])
{
	// Swap the elements
	int indx0 = rand_nums[0];
	int indx1 = rand_nums[1];
	std::swap(child_xcoord[indx0], child_xcoord[indx1]);
	std::swap(child_ycoord[indx0], child_ycoord[indx1]);
}

void execute(int pop_size,
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
	auto rgen = bind(uniform_real_distribution<>(0, 1), mt19937(rseed));

	// The fitness for the current generation
	const int individual_size = baseWorld.num_cities;

	// The best individuals
	int best_generation = 0;
	World bestLeader(baseWorld.num_cities, baseWorld.height, baseWorld.width);
	World generationLeader(baseWorld.num_cities, baseWorld.height, baseWorld.width);
	
	// Initialize the populations
	Population* oldPop = new Population(pop_size, baseWorld, seed);
	Population* newPop = new Population(pop_size, baseWorld.num_cities, baseWorld.height, baseWorld.width);
	
	// Calculate the fitnesses
	evaluate(*oldPop);
	
	// Initialize the best leader
	oldPop->select_leader(generationLeader, bestLeader);
	print_status(generationLeader, bestLeader, 0);
	gen_log.write_log(0, 0, generationLeader);

	// Continue through all generations
	for (int i = 0; i < max_gen; i++)
	{
		// Start the generation clock
		gen_clock = clock();

		// Create a new population
		for (int j = 0; j < pop_size; j++)
		{
			// Parents and children
			int* parents_xcoord[2];
			int* parents_ycoord[2];
			int* child_xcoord = new int[individual_size];
			int* child_ycoord = new int[individual_size];
			parents_xcoord[0] = new int[individual_size];
			parents_xcoord[1] = new int[individual_size];
			parents_ycoord[0] = new int[individual_size];
			parents_ycoord[1] = new int[individual_size];
			
			// Generate all probabilities ahead of time
			float prob_select[2] = {static_cast<float>(rgen()), static_cast<float>(rgen())};
			float prob_cross     = static_cast<float>(rgen());
			int   cross_loc      = static_cast<int>(rgen() * (baseWorld.num_cities - 1));
			float prob_mutate    = static_cast<float>(rgen());
			int   mutate_loc[2]  = {
				static_cast<int>(rgen() * baseWorld.num_cities),
				static_cast<int>(rgen() * baseWorld.num_cities)
			};
			while (mutate_loc[1] == mutate_loc[0])
				mutate_loc[1] = static_cast<int>(rgen() * baseWorld.num_cities);

			// Select two parents
			selection(*oldPop, parents_xcoord, parents_ycoord, prob_select);
			
			// Determine how many children are born
			if (prob_cross <= prob_crossover)
			{
				// Perform crossover
				crossover(parents_xcoord, parents_ycoord, child_xcoord, child_ycoord, baseWorld.num_cities, cross_loc);

				// Perform mutation
				if (prob_mutate <= prob_mutation)
					mutate(child_xcoord, child_ycoord, mutate_loc);

				// Add child to new population
				newPop->SetCities(j, child_xcoord, child_ycoord);
			}
			else // Select the first parent
			{
				// Perform mutation
				if (prob_mutate <= prob_mutation)
					mutate(parents_xcoord[0], parents_ycoord[0], mutate_loc);

				// Add child to new population
				newPop->SetCities(j, parents_xcoord[0], parents_ycoord[0]);
			}

			// Cleanup
			delete[] parents_xcoord[0]; delete[] parents_xcoord[1];
			delete[] parents_ycoord[0]; delete[] parents_ycoord[1];
			delete[] child_xcoord; delete[] child_ycoord;
		} // Population creation

		// Calculate the fitnesses
		evaluate(*newPop);

		// Swap the populations
		std::swap(oldPop, newPop);

		// Select the new leaders
		if (oldPop->select_leader(generationLeader, bestLeader))
			best_generation = i + 1;
		print_status(generationLeader, bestLeader, i + 1);
		gen_log.write_log(i + 1, end_clock(gen_clock), generationLeader);
	} // Generations
	
	delete oldPop; delete newPop;
	
	cout << endl
		 << "Best generation found at " << best_generation << " generations"
		 << endl;
}
