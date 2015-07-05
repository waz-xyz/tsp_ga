/* ga_gpu.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Header for GA implementation for the GPU
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/
//  Copyright (c) 2015 waz

#ifndef __GA_GPU_H__
#define __GA_GPU_H__

// Program includes
#include "world.h"
#include "log.h"

/*
	Runs the genetic algorithm on the GPU.
	
	pop_size       : The number of elements in the population
	max_gen        : The number of generations to run for
	prob_mutation  : The probability of a mutation occurring
	prob_crossover : The probability of a crossover occurring
	baseWorld      : The seed world, containing all of the desired cities
	gen_log        : A pointer a logger to be used for logging the generation statistics
	seed           : Seed for all random numbers
*/
void g_execute(int pop_size,
			   int max_gen,
			   float prob_mutation, float prob_crossover,
			   const World& baseWorld,
			   Logger& gen_log,
			   int seed);

#endif