/* world.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Header file for dealing with the 2D world.
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/
//  Copyright (c) 2015 waz

#ifndef __WORLD_H__
#define __WORLD_H__

// Native Includes
#include <iostream>
#include <cmath>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
////////// Shared
///////////////////////////////////////////////////////////////////////////////

struct City
{
	/*
		Stores the location of a city
	*/
	
	int x, y;
};

struct World
{
	/*
		2D world for the TSP
	*/
	
	int width, height; // World bounds
	int num_cities;    // Number of cities
	City* cities;      // Pointer to array of all of the cities
	float fitness;     // The current fitness
	float fit_prob;    // The fitness probability
	
	World()
	{
		cities = nullptr;
	}
	
	World(int num_cities, int height, int width)
	{
		init(num_cities, height, width);
	}
	
	/*
	 Makes a new world
	 
	 num_cities : The number of cities in the world
	 height     : The height of the world
	 width      : The width of the world
	 seed       : The random seed to use to select the cities
	 */
	World(int num_cities, int height, int width, int seed);
	
	~World();
	
	/*
	 Initialize a world struct
	 
	 num_cities : The number of cities in the world
	 height     : The height of the world
	 width      : The width of the world
	 */
	void init(int num_cities, int height, int width);

	void free();
	
	void calc_fitness();

	float calc_distance() const;
	
	/*
	 Initialize the population in host memory
	 
	 pop_size : The number of elements in the population
	 seed     : Seed for random number generation
	 
	 returns true if an error occurred
	 */
	World* initializePopulation(int pop_size, int seed) const;
	
	/*
	 Clones a single world in host memory
	 
	 src : Pointer to source world
	 */
	World& operator=(const World& src);
};

///////////////////////////////////////////////////////////////////////////////
////////// CPU functions
///////////////////////////////////////////////////////////////////////////////

/*
	Clones one more cities in host memory
	
	src        : Pointer to source cities
	dst        : Pointer to destination cities
	num_cities : The number of cities to clone
*/
void clone_cities(City* src, City* dst, int num_cities);

#endif