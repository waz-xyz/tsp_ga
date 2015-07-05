/* common.h

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Header for all shared functions
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/
//  Copyright (c) 2015 waz

#ifndef __COMMON_H__
#define __COMMON_H__

#include <fstream>
#include <ctime>

#include "g_type.h"

/*
	Stops a clocks timer and returns the elapsed time in ms.
	
	clk - The clk to work with.
*/
float end_clock(clock_t clk);

/*
	Utility function for dumping a GPU buffer to a file
*/
template<typename T>
void printBuffer(const opencl_env& env, const cl::Buffer buf, int length, const char* filename)
{
	T* values = new T[length];
	env.queue().enqueueReadBuffer(buf, CL_TRUE, 0, length*sizeof(T), values);
	std::ofstream outfile;
	outfile.open(filename);
	for (int i = 0; i < length; i++) {
		outfile << values[i] << std::endl;
	}
	outfile.close();
	delete[] values;
}

#endif