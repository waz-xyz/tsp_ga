//
//  population.cpp
//  tsp_ga
//
//  Created by waz on 19/06/15.
//  Copyright (c) 2015 waz
//

#include "population.h"
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>

Population::Population(int numIndividuals, int numCitiesPerWorld, int height, int width)
{
	this->numIndividuals = numIndividuals;
	this->height = height;
	this->width = width;
	this->numCitiesPerWorld = numCitiesPerWorld;
	int totalCities = numCitiesPerWorld * numIndividuals;
	this->cities_xcoord = new int[totalCities];
	this->cities_ycoord = new int[totalCities];
	this->fitness = new float[numIndividuals];
	this->fit_prob = new float[numIndividuals];
}

Population::Population(int numIndividuals, const World& baseWorld, int seed)
{
	this->numIndividuals = numIndividuals;
	this->height = baseWorld.height;
	this->width = baseWorld.width;
	this->numCitiesPerWorld = baseWorld.num_cities;
	int totalCities = numCitiesPerWorld * numIndividuals;
	this->cities_xcoord = new int[totalCities];
	this->cities_ycoord = new int[totalCities];
	this->fitness = new float[numIndividuals];
	this->fit_prob = new float[numIndividuals];
	
	// Set the seed for random number generation
	srand(seed);
	
	City* cities = new City[numCitiesPerWorld];
	memmove(cities, baseWorld.cities, numCitiesPerWorld*sizeof(City));
	
	for (int i = 0; i < numIndividuals; i++)
	{
		int baseOffset = i * numCitiesPerWorld;
		random_shuffle(cities, &cities[numCitiesPerWorld]);
		
		for (int j = 0; j < numCitiesPerWorld; j++) {
			cities_xcoord[baseOffset + j] = cities[j].x;
			cities_ycoord[baseOffset + j] = cities[j].y;
		}
	}
	
	delete[] cities;
}

Population::~Population()
{
	delete[] cities_xcoord;
	delete[] cities_ycoord;
}

float Population::CalcFitness(int indx)
{
	/*
	 Evaluates the fitness function
		*/
	
	assert(0 <= indx && indx < numIndividuals);
	
	int baseOffset = indx*numCitiesPerWorld;
	int distance = 0;
	for (int i = 0; i < numCitiesPerWorld - 1; i++) {
		int dy = cities_ycoord[baseOffset + i] - cities_ycoord[baseOffset + i + 1];
		int dx = cities_xcoord[baseOffset + i] - cities_xcoord[baseOffset + i + 1];
		
		distance += dx*dx + dy*dy;
	}
	return fitness[indx] = (width * height) / static_cast<float>(distance);
}

void Population::GetWorld(World& world, int inx) const
{
	assert(0 <= inx && inx < numIndividuals);
	
	world.num_cities = this->numCitiesPerWorld;
	world.height     = this->height;
	world.width		 = this->width;
	world.fitness	 = this->fitness[inx];
	world.fit_prob	 = this->fit_prob[inx];
	if (world.cities == nullptr)
	{
		world.cities = new City[numCitiesPerWorld];
	}
	int baseOffset = inx * numCitiesPerWorld;
	for (int i = 0; i < numCitiesPerWorld; i++) {
		world.cities[i].x = cities_xcoord[baseOffset + i];
		world.cities[i].y = cities_ycoord[baseOffset + i];
	}
}

void Population::GetCities(int* cities_xcoord, int* cities_ycoord, int inx) const
{
	memmove(cities_xcoord, &(this->cities_xcoord[inx*numCitiesPerWorld]), numCitiesPerWorld*sizeof(int));
	memmove(cities_ycoord, &(this->cities_ycoord[inx*numCitiesPerWorld]), numCitiesPerWorld*sizeof(int));
}

void Population::SetCities(int inx, int* cities_xcoord, int* cities_ycoord)
{
	memmove(&(this->cities_xcoord[inx*numCitiesPerWorld]), cities_xcoord, numCitiesPerWorld*sizeof(int));
	memmove(&(this->cities_ycoord[inx*numCitiesPerWorld]), cities_ycoord, numCitiesPerWorld*sizeof(int));
}

int Population::select_leader(World& generationLeader, World& bestLeader) const
{
	/*
		Updates the generation and global best leaders
		
		pop               : The population to select from
		pop_size          : The number of elements in the population
		generation_leader : The world with the max fitness for this generation
		best_leader       : The world with the best global fitness across all generations
	 
		return 1 if this generation is the best, else 0
	 */
	
	// Find element with the largest fitness function
	int ix = 0;
	for (int i = 1; i < numIndividuals; i++) {
		if (fitness[i] > fitness[ix])
			ix = i;
	}
	
	// Store generation leader
	GetWorld(generationLeader, ix);
	
	// Update best leader
	if (generationLeader.fitness > bestLeader.fitness) {
		bestLeader = generationLeader;
		return 1;
	}
	
	return 0;
}
