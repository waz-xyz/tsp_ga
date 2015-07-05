//
//  g_population.h
//  tsp_ga
//
//  Created by waz on 20/06/15.
//  Copyright (c) 2015 waz
//

#ifndef __tsp_ga__g_population__
#define __tsp_ga__g_population__

#include "g_type.h"
#include "world.h"

class g_Population
{
private:
	const opencl_env& env;
	int numIndividuals;
	int numCitiesPerWorld;
	int height, width;
	cl::Buffer cities_xcoord;
	cl::Buffer cities_ycoord;
	cl::Buffer fitness;
	cl::Buffer fit_prob;
	void extract_max_fit(World& world) const;
public:
	g_Population(const opencl_env& env, int numIndividuals, int numCitiesPerWorld, int height, int width);
	g_Population(const opencl_env& env, int numIndividuals, const World& baseWorld, int seed);
	void evaluate();
	int select_leader(World& generation_leader, World& best_leader) const;
	void select_parents(cl::Buffer& selected_parents_inx, cl::Buffer& probs) const;
	void next_generation(g_Population& new_pop,
						 const cl::Buffer& d_sel_ix,
						 float prob_crossover, const cl::Buffer d_prob_cross, const cl::Buffer d_cross_loc,
						 float prob_mutation, const cl::Buffer d_prob_mutate, const cl::Buffer d_mutate_loc) const;
};

#endif /* defined(__tsp_ga__g_population__) */
