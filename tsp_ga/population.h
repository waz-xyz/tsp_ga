//
//  population.h
//  tsp_ga
//
//  Created by waz on 19/06/15.
//  Copyright (c) 2015 waz
//

#ifndef __tsp_ga__population__
#define __tsp_ga__population__

#include "world.h"

struct Population
{
	int numIndividuals;
	int numCitiesPerWorld;
	int height, width;
	int *cities_xcoord;
	int *cities_ycoord;
	float *fitness;
	float *fit_prob;
	
	Population(int numIndividuals, int numCitiesPerWorld, int height, int width);
	Population(int numIndividuals, const World& baseWorld, int seed);
	~Population();
	float CalcFitness(int indx);
	void GetWorld(World& world, int inx) const;
	void GetCities(int* cities_xcoord, int* cities_ycoord, int inx) const;
	void SetCities(int inx, int* cities_xcoord, int* cities_ycoord);
	
	/*
	 Updates the generation and global best leaders
	 
	 generation_leader : The world with the max fitness for this generation
	 best_leader       : The world with the best global fitness across all generations
	 
	 return 1 if this generation is the best, else 0
	 */
	int select_leader(World& generationLeader, World& bestLeader) const;
};


#endif /* defined(__tsp_ga__population__) */
