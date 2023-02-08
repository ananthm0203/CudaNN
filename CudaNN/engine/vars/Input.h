#pragma once

#include "../base/tensor.h"
#include "initializers/initializers.h"

#include <vector>

class Input
{
public:

	Input(const Shape& shape)
		: tensor(shape, Tensor::LayerType::Input)
	{
	};

	Input(const Shape& shape, const std::vector<float>& vals)
		: tensor(shape, Tensor::LayerType::Input)
	{
		assert(vals.size() == tensor.get_shape().size);
		memcpy(tensor.raw(), &vals[0], vals.size());
	}

	Input(const Shape& shape, Initializer& init)
		: tensor(shape, Tensor::LayerType::Input)
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
