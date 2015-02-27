#pragma once

#include <iostream>
#include <exception>
#include <vector>
#include <numeric>

#include <stdint.h>

void getHist(const std::vector<int>& shape, int*& data)
{
	// Create array from dimension description
	int tSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	int32_t* array = new int32_t[tSize];

	//Fill some values
	for(unsigned int i = 0; i < tSize; ++i)
		array[i] = i;

	// Set result
	data = array;
}
