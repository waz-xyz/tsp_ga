/* ga_cpu.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Header for GA implementation for the CPU
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/
//  Copyright (c) 2015 waz

#ifndef __GA_CPU_H__
#define __GA_CPU_H__

// Program includes
#include "world.h"
#include "population.h"
#include "log.h"

/*
	Evaluate the fitness function and calculate the
	fitness probabilities.
*/
void evaluate(Population& pop);

/*
	Perform the selection algorithm on the CPU.
	This selection algorithm uses Roulette Wheel Selection.
	Two parents will be selected at a time, from the population.

	pop       : The population to select from
	parents_* : The cities for two worlds
	rand_nums : The random numbers to use
*/
void selection(const Population& pop, int* parent_xcoord[], int* parent_ycoord[], float rand_nums[2]);

/*
	Perform the crossover algorithm on the CPU.
	This crossover algorithm uses the Single Point Crossover method.
	
	parents_*  : The cities for two worlds
	child_*    : The child to create
	num_cities : The number of cities in the world
	cross_over : The location to perform crossover
*/
void crossover(int* parents_xcoord[2], int* parents_ycoord[2], int* child_xcoord, int* child_ycoord, int num_cities, int cross_over);

/*
	Perform the mutation algorithm on the CPU.
	This mutation algorithm uses the order changing permutation method.
	
	child_*    : The child to mutate
	rand_nums  : The random numbers to use
*/
void mutate(int* child_xcoord, int* child_ycoord, int rand_nums[2]);

/*
	Runs the genetic algorithm on the CPU.
	
	pop_size       : The number of elements in the population
	max_gen        : The number of generations to run for
	prob_mutation  : The probability of a mutation occurring
	prob_crossover : The probability of a crossover occurring
	baseWorld      : The seed world, containing all of the desired cities
	gen_log        : A pointer a logger to be used for logging the generation statistics
	seed           : Seed for all random numbers
*/
void execute(int pop_size,
			 int max_gen,
			 float prob_mutation, float prob_crossover,
			 const World& baseWorld,
			 Logger& gen_log,
			 int seed);

#endif