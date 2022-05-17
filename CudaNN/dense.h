#ifndef DENSE_H
#define DENSE_H
#include "layer.h"

class Dense :
	public Layer
{
	explicit Dense(size_t channelIn, size_t channelOut, bool useBias = true, unique_ptr<Activation> activation = nullptr) :
		Layer(channelIn, channelOut, useBias, std::move(activation)) {};
	~Dense() = default;
	void operator()(std::unique_ptr<float> X, std::unique_ptr<float> dest); // To be done on CUDA
	void backward(std::unique_ptr<float> X, std::unique_ptr<float> dest); // To be done on CUDA
};
#endif // DENSE_H

