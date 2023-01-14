#pragma once

#include "../op.h"

class Relu
	: public Op
{
public:

	Tensor* operator()(Tensor* in)
	{
		this->in = in;
		out = Tensor(in->get_shape(), Tensor::LayerType::Output);
		return &out;
	}

	void forwards();
	void backwards();

private:

	Tensor* in;
	Tensor out;
};