//  Copyright (c) 2014 James Mnatzaganian
//  Copyright (c) 2015 waz

//
// Evaluates the fitness function
//
__kernel void fitness(int pop_len,
					  int num_cities,
					  int WxH,
					  __global int* x_coord,
					  __global int* y_coord,
					  __global float* fitness)
{
	int tid = get_global_id(0);
	
	if (tid < pop_len) {
		int distance = 0; // Total "normalized" "distance"
		
		// Calculate fitnesses
		int baseOffset = tid * num_cities;
		for (int i = 0; i < num_cities-1; i++) {
			int dx = x_coord[baseOffset + i] - x_coord[baseOffset + i + 1];
			int dy = y_coord[baseOffset + i] - y_coord[baseOffset + i + 1];
			distance += dx*dx + dy*dy;
		}
		
		fitness[tid] = (float)WxH / (float)distance;
	}
}

//
// Calculation of fitness probabilities
// Step 1: Calculate the partial sums of fitness values
//
__kernel void fit_sum(int pop_len,
					  __global float* fitness,
					  __global float* fit_prob,
					  __global float* accumFit)
{
	int tid = get_global_id(0);
	
	if (tid < pop_len) {
		// Sum of all fitness
		float sum = 0.0f;
			
		// Calculate the partial sum
		for (int i = 0; i <= tid; i++)
			sum += fitness[i];
		fit_prob[tid] = sum;
		
		// Copy over the final result
		if (tid == (pop_len - 1))
			*accumFit = sum;
	}
}

//
// Calculation of fitness probabilities
// Step 2: Divide the sum of the previous kernel between the total sum of
//         fitness values
//
__kernel void fit_prob(int pop_len,
					   __global float* fit_prob,
					   float fit_sum)
{
	int tid = get_global_id(0);
	if (tid < pop_len) {
		fit_prob[tid] /= fit_sum;
	}
}

//
// Finds the maximum fitness value
// Step 1
//
__kernel void max_fit_phase_0(int pop_len,
							  __global float* fitness,
							  __local float* scratch_val,
							  __local int* scratch_inx,
							  __global float* result_val,
							  __global int* result_inx)
{
	int ginx = get_global_id(0);
	int max_index;
	float accumulator = -INFINITY;
	
	while (ginx < pop_len)	{
		float element = fitness[ginx];
		if (element > accumulator) {
			accumulator = element;
			max_index = ginx;
		}
		ginx += get_global_size(0);
	}
	
	int linx = get_local_id(0);
	scratch_val[linx] = accumulator;
	scratch_inx[linx] = max_index;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Apply reduction
	for (int offset = get_local_size(0) / 2;
		 offset > 0;
		 offset /= 2)
	{
		if (linx < offset) {
			float other = scratch_val[linx + offset];
			float mine = scratch_val[linx];
			if (other > mine) {
				scratch_val[linx] = other;
				scratch_inx[linx] = scratch_inx[linx + offset];
			}
			
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (linx == 0) {
		int gid = get_group_id(0);
		result_val[gid] = scratch_val[0];
		result_inx[gid] = scratch_inx[0];
	}
}

//
// Finds the maximum fitness value
// Step 2
// Note: It assumes that 'length' is very small (and so doesn't incur into
//       another reduction)
//
__kernel void max_fit_phase_1(int length,
							  __global float* result_val,
							  __global int* result_inx)
{
	int tid = get_global_id(0);
	if (tid == 0) {
		float maxval = result_val[0];
		int maxinx = result_inx[0];
		
		for (int i = 1; i < length; i++) {
			float x = result_val[i];
			if (x > maxval) {
				maxval = x;
				maxinx = result_inx[i];
			}
		}
		result_inx[0] = maxinx;
	}
}

//
// Finds the indexes of the selected parents
//
__kernel void select_parents(int pop_size,
							 __global float* fit_prob,
							 __global float* rand_nums,
							 __global int* sel_ix)
{
	int tid = get_global_id(0);
	
	if (tid < (2 * pop_size)) {
		for (int i = 0; i < pop_size; i++) {
			if (rand_nums[tid] < fit_prob[i]) {
				sel_ix[tid] = i;
				break;
			}
		}
	}
}

//
// Performs the crossover operation
//
__kernel void crossover(int pop_len,
						int num_cities,
						__global const int* old_x_coord,
						__global const int* old_y_coord,
						__global int* new_x_coord,
						__global int* new_y_coord,
						__global int* selected_parents_inx,
						float prob_crossover,
						__global float* rnd_prob_cross,
						__global int* cross_loc)
{
	int tid = get_global_id(0);
	
	if (tid < pop_len) {
		if (rnd_prob_cross[tid] < prob_crossover) {
			int cross_location = cross_loc[tid];
			
			// Copy elements from first parent up through crossover point
			int parent_0_loc = selected_parents_inx[2*tid];
			int old_base_offset = parent_0_loc * num_cities;
			int new_base_offset = tid * num_cities;
			
			for (int i = 0; i <= cross_location; i++) {
				new_x_coord[new_base_offset + i] = old_x_coord[old_base_offset + i];
				new_y_coord[new_base_offset + i] = old_y_coord[old_base_offset + i];
			}
			
			// Add remaining elements from second parent to child, in order
			int remaining = num_cities - cross_location - 1;
			int count = 0;
			int parent_1_loc = selected_parents_inx[2 * tid + 1];
			old_base_offset = parent_1_loc * num_cities;
			new_base_offset = tid * num_cities;
			
			for (int i = 0; i < num_cities; i++) {  // Loop parent
				bool in_child = false;
				
				for (int j = 0; j <= cross_location; j++) {    // Loop child
					// If the city is in the child, exit
					if (new_x_coord[new_base_offset + j] == old_x_coord[old_base_offset + i] &&
						new_y_coord[new_base_offset + j] == old_y_coord[old_base_offset + i])
					{
						in_child = true;
						break;
					}
				}
				
				// If the city was not found in the child, add it to the child
				if (!in_child) {
					count++;
					new_x_coord[new_base_offset + cross_location + count] = old_x_coord[old_base_offset + i];
					new_y_coord[new_base_offset + cross_location + count] = old_y_coord[old_base_offset + i];
				}
				
				// Stop once all of the cities have been added
				if (count == remaining) break;
			}
		}
	}
}

//
// Clones a parent when a crossover is not applied
//
__kernel void clone_parent(int pop_len,
						   int num_cities,
						   __global const int* old_x_coord,
						   __global const int* old_y_coord,
						   __global int* new_x_coord,
						   __global int* new_y_coord,
						   float prob_crossover,
						   __global const float* rnd_prob_cross,
						   __global const int* selected_parents_inx)
{
	int tid = get_global_id(0);
	
	if (tid < pop_len) {
		if (rnd_prob_cross[tid] >= prob_crossover) {
			int loc = selected_parents_inx[2*tid];
			int old_base_offset = loc * num_cities;
			int new_base_offset = tid * num_cities;
			
			for (int i = 0; i < num_cities; i++) {
				new_x_coord[new_base_offset + i] = old_x_coord[old_base_offset + i];
				new_y_coord[new_base_offset + i] = old_y_coord[old_base_offset + i];
			}
		}
	}
}

//
// Performs the mutation operation
//
__kernel void mutate(int pop_len,
					 int num_cities,
					 __global int* x_coord,
					 __global int* y_coord,
					 float prob_mutation,
					 __global const float* rnd_prob_mutation,
					 __global const int* rnd_mutate_loc)
{
	int tid = get_global_id(0);
	
	if (tid < pop_len) {
		if (rnd_prob_mutation[tid] < prob_mutation) {
			int loc0 = rnd_mutate_loc[2*tid];
			int loc1 = rnd_mutate_loc[2*tid+1];
			int offset0 = tid*num_cities + loc0;
			int offset1 = tid*num_cities + loc1;
			
			int tmp = x_coord[offset0];
			x_coord[offset0] = x_coord[offset1];
			x_coord[offset1] = tmp;
			
			tmp = y_coord[offset0];
			y_coord[offset0] = y_coord[offset1];
			y_coord[offset1] = tmp;
		}
	}
}

