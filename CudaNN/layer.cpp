#include <cuda.h>
#include <cuda_runtime.h>
#include "layer.h"

Layer::Layer(size_t channelIn, size_t channelOut, bool useBias, unique_ptr<Activation> activation)
	: channelIn(channelIn), channelOut(channelOut), activation(std::move(activation))
{
	weights = make_unique<float>(channelIn * channelOut);
	if (useBias)
	{
		bias = make_unique<float>(channelOut);
	}
	else
	{
		bias = nullptr;
	}
}
