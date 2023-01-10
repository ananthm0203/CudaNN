#pragma once

#include "../op.h"

#include <vector>


class Add : public Op
{
public:

	Tensor* operator()(std::vector<Tensor*> inputs)
	{
		assert(inputs.size() >= 2);
		auto& out_shape = inputs[0]->get_shape();
		for (size_t i = 1; i < inputs.size(); ++i)
		{
			assert(inputs[i]->get_shape() == out_shape);
		}
		handle_inputs(inputs);
		ins = inputs;
		out = Tensor(out_shape, Tensor::LayerType::Output);
		return &out;
	}

	void forward();
	void backwards()
	{
		auto& in_grad = out.gradient();
		for (auto& in : ins)
		{
			in->update_gradient(this, in_grad);
		}
	}

private:

	std::vector<Tensor*> ins;
	Tensor out;
};