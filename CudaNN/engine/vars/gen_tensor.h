#pragma once

#include "../base/tensor.h"
#include "initializers/initializers.h"

class GenTensor
{
public:

	GenTensor(const Shape& shape)
		: tensor(shape, Tensor::LayerType::Weight)
	{
		ZeroInitializer zi;
		tensor.initialize(zi);
	};

	GenTensor(const Shape& shape, Initializer& init)
		: tensor(shape, Tensor::LayerType::Weight)
	{
		tensor.initialize(init);
	}

	Tensor* operator()()
	{
		return &tensor;
	}

private:

	Tensor tensor;
};
