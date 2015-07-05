/* common.cpp

   Author        : James Mnatzaganian
   Date Created  : 11/07/14
   
   Description   : Module for implementing functions shared between the CPU and
   GPU implementations.
   License       : MIT License http://opensource.org/licenses/mit-license.php
   Copyright     : (c) 2014 James Mnatzaganian
*/
//  Copyright (c) 2015 waz

// Native includes
#include <iostream>
#include <ctime>

// Program includes
#include "common.h"

float end_clock(clock_t clk)
{	
	return ((float)((clock() - clk) * 1000) / (float)CLOCKS_PER_SEC);
}
