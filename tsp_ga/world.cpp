/* world.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Module for dealing with the TSP's world
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/
//  Copyright (c) 2015 waz

// Native Includes
#include <set>
#include <tuple>
#include <random>
#include <functional>
#include <cstring>

// Program Includes
#include "world.h"
#include "common.h"

World::World(int num_cities, int height, int width, int seed)
{
	// Random number generation
	mt19937::result_type rseed = seed;
	auto rgen = bind(uniform_real_distribution<>(0, 1), mt19937(rseed));
	
	// Initialize the world
	init(num_cities, height, width);
	
	// Create a set to deal with uniqueness
	set<tuple<int, int>> coordinates;
	
	// Create some unique random cities
	for (int i = 0; i < num_cities; i++)
	{
		while (true)
		{
			// Try to add a new set of coordinates
			int rwidth = static_cast<int>(rgen() * width);
			int rheight = static_cast<int>(rgen() * height);
			tuple<int,int> coors(rwidth, rheight);
			auto ret = coordinates.insert(coors);
			
			// Break if the city was added successfully
			if (ret.second)
				break;
		}
	}
	
	// Add those cities to the world
	{
		int i = 0;
		for (auto it = coordinates.begin(); it!=coordinates.end(); it++)
		{
			this->cities[i].x = get<0>(*it);
			this->cities[i].y = get<1>(*it);
			i++;
		}
	}
}

World::~World()
{
	free();
}

void World::init(int num_cities, int height, int width)
{
	this->width      = width;
	this->height     = height;
	this->num_cities = num_cities;
	this->fitness    = 0.0f;
	this->fit_prob   = 0.0f;
	this->cities     = new City[num_cities];
}

void World::free()
{
	if (cities != nullptr)
		delete[] cities;
}

void World::calc_fitness()
{
	/*
	 Evaluates the fitness function
		*/
	
	int distance = 0;
	for (int i = 0; i < num_cities - 1; i++) {
		int dx = cities[i].x - cities[i + 1].x;
		int dy = cities[i].y - cities[i + 1].y;
		distance += dx*dx + dy*dy;
	}
	this->fitness = (width * height) / static_cast<float>(distance);
}

float World::calc_distance() const
{
	/*
	 Calculates the distance travelled
		*/
	
	float distance = 0.0f;
	for (int i = 0; i < num_cities - 1; i++) {
		int dx = cities[i].x - cities[i + 1].x;
		int dy = cities[i].y - cities[i + 1].y;
		distance += sqrtf(dx*dx + dy*dy);
	}
	return distance;
}

World* World::initializePopulation(int pop_size, int seed) const
{
	/*
		Allocate and initialize the population in host memory
	 
		pop_size : The number of elements in the population
		seed     : Seed for random number generation
	 */
	
	World* pop = new World[pop_size];
	
	// Set the seed for random number generation
	srand(seed);
	
	for (int i=0; i<pop_size; i++)
	{
		// Clone world
		pop[i] = *this;
		
		// Randomly adjust the path between cities
		random_shuffle(&pop[i].cities[0], &pop[i].cities[this->num_cities]);
	}
	
	return pop;
}

World& World::operator=(const World& src)
{
	this->width      = src.width;
	this->height     = src.height;
	this->num_cities = src.num_cities;
	this->fitness    = src.fitness;
	this->fit_prob   = src.fit_prob;
	if (this->cities == nullptr)
	{
		this->cities = new City[src.num_cities];
	}
	clone_cities(src.cities, this->cities, src.num_cities);
	
	return *this;
}

void clone_cities(City* src, City* dst, int num_cities)
{
	/*
		Clones one more cities in host memory
		
		src        : Pointer to source cities
		dst        : Pointer to destination cities
		num_cities : The number of cities to clone
	*/
	
	memcpy(dst, src, num_cities * sizeof(City));
}

