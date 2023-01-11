#pragma once

#include "../base/tensor.h"
#include "../gradient/gradient.h"
#include "../ops/op.h"
#include "initializers/initializers.h"

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
		ZeroInitializer zi;
		tensor.initialize(zi);
	};

	GenTensor(const Shape& shape, InitMethod im, Initializer& init)
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
