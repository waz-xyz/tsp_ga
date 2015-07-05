//
//  type.h
//  tsp_ga
//
//  Created by waz on 20/06/15.
//  Copyright (c) 2015 waz
//

#ifndef tsp_ga_g_type_h
#define tsp_ga_g_type_h

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#	include <OpenCL/cl.hpp>
#else
#	include <CL/cl.hpp>
#endif

enum class kernel_t
{
	fitness,
	fit_sum,
	fit_prob,
	max_fit_phase_0,
	max_fit_phase_1,
	select_parents,
	crossover,
	clone_parent,
	mutate,
	LENGTH = mutate+1
};

class opencl_env
{
private:
	cl::Kernel* krnl_table[static_cast<int>(kernel_t::LENGTH)];
	std::vector<cl::Platform> platforms;
	cl::Context* _context;
	std::vector<cl::Device> devices;
	cl::CommandQueue _queue;
	cl::Program* program;
	cl::NDRange globalRange;
	int numComputeUnits;
public:
	opencl_env();
	~opencl_env();
	
	cl::Context& context() const {
		return *_context;
	}
	
	const cl::CommandQueue& queue() const {
		return _queue;
	}
	
	int getNumComputeUnits() const {
		return numComputeUnits;
	}
	
	cl::Kernel& getKernel(kernel_t) const;
};

#endif
