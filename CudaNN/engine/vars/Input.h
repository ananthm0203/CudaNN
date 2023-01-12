#pragma once

#include "../base/tensor.h"
#include "initializers/initializers.h"

class Input
{
public:

	Input(const Shape& shape)
		: tensor(shape, Tensor::LayerType::Input)
	{
	};

	Tensor* operator()()
	{
		return &tensor;
	}

private:
	Tensor tensor;
};
