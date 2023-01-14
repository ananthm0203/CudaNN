#pragma once

#include "../op.h"

class Softmax
	: public Op
{
public:

	Tensor* operator()(Tensor* in)
	{
		assert(in->get_shape().H == in->get_shape().W == 1 && in->get_shape().W <= 1024);
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
