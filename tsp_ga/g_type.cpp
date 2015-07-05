//
//  g_type.cpp
//  tsp_ga
//
//  Created by waz on 20/06/15.
//  Copyright (c) 2015 waz
//

#include "g_type.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

static const char *kernelsrcpath { "kernel.cl" };

opencl_env::opencl_env()
{
	try {
		cl::Platform::get(&platforms);
		
		static cl_context_properties cps[3] = {
			CL_CONTEXT_PLATFORM, cl_context_properties(platforms[0]()), 0
		};
		
		_context = new cl::Context(CL_DEVICE_TYPE_GPU, cps);
		devices = _context->getInfo<CL_CONTEXT_DEVICES>();
		_queue = cl::CommandQueue(context(), devices[0], 0);
		
		std::ifstream kernelSrcFile(kernelsrcpath);
		std::string kernelSrcText((std::istreambuf_iterator<char>(kernelSrcFile)), std::istreambuf_iterator<char>());
		
		cl::Program::Sources kernelSrc(1, std::make_pair(kernelSrcText.c_str(), kernelSrcText.length()+1));
		program = new cl::Program(context(), kernelSrc);
		
		try {
			program->build(devices);
		} catch (cl::Error error) {
			if (strcmp(error.what(), "clBuildProgram") == 0) {
				std::cerr << "Error while building:" << std::endl;
				std::cerr << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
				exit(1);
			} else {
				throw;
			}
		}
		
		numComputeUnits = devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		
		krnl_table[static_cast<int>(kernel_t::fitness)] = new cl::Kernel(*program, "fitness");
		krnl_table[static_cast<int>(kernel_t::fit_sum)] = new cl::Kernel(*program, "fit_sum");
		krnl_table[static_cast<int>(kernel_t::fit_prob)] = new cl::Kernel(*program, "fit_prob");
		krnl_table[static_cast<int>(kernel_t::max_fit_phase_0)] = new cl::Kernel(*program, "max_fit_phase_0");
		krnl_table[static_cast<int>(kernel_t::max_fit_phase_1)] = new cl::Kernel(*program, "max_fit_phase_1");
		krnl_table[static_cast<int>(kernel_t::select_parents)] = new cl::Kernel(*program, "select_parents");
		krnl_table[static_cast<int>(kernel_t::crossover)] = new cl::Kernel(*program, "crossover");
		krnl_table[static_cast<int>(kernel_t::clone_parent)] = new cl::Kernel(*program, "clone_parent");
		krnl_table[static_cast<int>(kernel_t::mutate)] = new cl::Kernel(*program, "mutate");
	} catch (cl::Error error) {
		std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
		exit(1);
	}
}

opencl_env::~opencl_env()
{
	for (int i = 0; i < static_cast<int>(kernel_t::LENGTH); i++) {
		delete krnl_table[i];
	}
	delete program;
	delete _context;
}

cl::Kernel& opencl_env:: getKernel(kernel_t id) const
{
	return *krnl_table[static_cast<int>(id)];
}
