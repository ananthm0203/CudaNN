#pragma once

#include "../base/tensor.h"
#include "../gradient/gradient.h"
#include "../ops/op.h"

class GenTensor
{
public:

	enum class InitMethod
	{
		Zero,
		GlorotNormal,
		GlorotUniform,
		HeNormal,
		HeUniform
	};

	GenTensor(const Shape& shape, InitMethod im)
		: tensor(shape, Tensor::LayerType::Weight)
	{
		// TODO: initialization
	};

private:
	Tensor tensor;
};
