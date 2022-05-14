#ifndef DENSE_H
#define DENSE_H
#include "layer.h"

class Dense :
	public Layer
{
	explicit Dense(size_t channelIn, size_t channelOut, bool useBias = true) :
		Layer(channelIn, channelOut, useBias) {};
	~Dense() = default;
	float* forward(float* x); // To be done on CUDA
	float* backward(); // To be done on CUDA
};
#endif // DENSE_H

