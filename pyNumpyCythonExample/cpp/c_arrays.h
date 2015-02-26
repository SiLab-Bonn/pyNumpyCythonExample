#pragma once

#include <iostream>
#include <exception>

void getHist(int*& data, unsigned int& size)
{
	const int tSize = 16;
	int* array = new int[tSize];
	for (int i = 0; i < tSize; ++i)
		array[i] = i;
	data = array;
	size = tSize;
}